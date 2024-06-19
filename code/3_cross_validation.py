import jsonlines
import json
import random
import re
import os
import copy
import nltk
import numpy as np
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)
import logging
import signal
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError


random.seed(0)


# test gpt4

os.environ['NLTK_DATA'] = 'your nltk_data data path'
logging.getLogger('nltk').setLevel(logging.CRITICAL)
from nltk import data
data.path.append('your nltk_data data path')

path="./sample_data/eval_func_rft.jsonl"


results = list(jsonlines.open(path))


print("Preprocess vertification functions")


collect_packages = []
for result in results:
    res = result['gpt-answer']
    eval_funcs, test_cases = [], []
    for each in res:
        try:
            json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
        except IndexError:
            continue
    
    # func rejection
    try:
        res_dict = json.loads(json_dict)
    except json.JSONDecodeError:
        continue
        
    func = res_dict['func']
    
    if '\\n' in func:
        func = func.replace('\\n', '\n')
    try:
        exec(func)
    except Exception:
        continue
    
    for line in func.split('\n'):
        if 'import' in line or 'download' in line or 'requests' in line:
            collect_packages.append(line)
print(list(set(collect_packages)))





print("cross validation for functions and cases")

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

filter_results = []
for result in tqdm(results):
    res = result['gpt-answer']
    eval_funcs, test_cases = [], []
    for each in tqdm(res):
        try:
            json_dict = re.findall(r'```json(.*?)```', each, re.DOTALL)[0].strip()
        except IndexError:
            continue

        try:
            res_dict = json.loads(json_dict)
        except json.JSONDecodeError:
            continue

        # func rejection
        func = res_dict['func']
        func = func.strip()
        func = '\n'.join([each for each in func.split('\n') if 'download' not in each and 'requests' not in each])
        try:
            exec(func)
        except Exception:
            continue
        eval_funcs.append(func)

        for each in res_dict['cases']:
            try:
                test_cases.append((each['input'], each['output']))
            except KeyError:
                print(each)
    eval_funcs = list(set(eval_funcs))
    test_cases = list(map(json.loads, set(map(json.dumps, test_cases))))
    if len(eval_funcs) < 3 or len(test_cases) < 10:
        continue

    filtered_test_cases = []

    for each in tqdm(test_cases):
  

        flag = False
        for func in eval_funcs:
            local_vars = {}
 
            try:
                exec(func, globals(), local_vars)
            except Exception:
                continue
            
            if 'evaluate' not in local_vars:
                continue
            eval_func = local_vars['evaluate']
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                res = eval_func(each[0])
            except Exception:
                res = None
            finally:
                signal.alarm(0)
            if res is not None and res == each[1]:
                flag = True
        if flag:
            filtered_test_cases.append(each)

    scored_funcs = []
    for func in tqdm(eval_funcs):
        local_vars = {}
        try:
            exec(func, globals(), local_vars)
        except Exception:
                continue
        if 'evaluate' not in local_vars:
            continue

        eval_func = local_vars['evaluate']
        acc = []
        for inp, out in filtered_test_cases:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                res = eval_func(inp)
            except Exception:
                res = None
            finally:
                signal.alarm(0)
            if res is None or res != out:
                acc.append(0)
            else:
                acc.append(1)
        acc = np.mean(acc) if acc else 0
        scored_funcs.append((func, acc))

    valid_funcs = [each for each in scored_funcs if each[1] >= 0.8]
    if not valid_funcs:
        continue

    filter_results.append({
        "instruction": result['instruction'],
        "eval_func": valid_funcs,
        "cases": filtered_test_cases
    })
    

print("finish!!!")

with jsonlines.open("./output/cross_validation.jsonl", "w") as f:
    for each in filter_results:
        f.write(each)