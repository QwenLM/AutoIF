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


'''
Please process your SFT data with the generated eval functions in previous step, as the format in dpo_query_eval_score_results.jsonl
'''

results = list(jsonlines.open("./sample_data/query_w_funcs.jsonl"))


def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

count=0
filter_samples = []
for result in tqdm(results):
    eval_funcs = []
    

    for func, score in result['eval_func']:
        local_vars = {}
        try:
            exec(func, globals(), local_vars)
        except Exception:
                continue
        eval_funcs.append(local_vars['evaluate'])
    
    
    filter_responses = []

    for response in result['gpt-answer']:
        acc = []
        for eval_func in eval_funcs:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(3)
                res = eval_func(response)
            except Exception as e:
                print(e)
                res = None
            finally:
                signal.alarm(0)
            
            if res is not None:
                try:
                    acc.append(int(res))
                except:
                    continue
        acc = np.mean(acc) if acc else 0

        filter_responses.append([response,acc])
    
   
    try:
        filter_samples.append({
            'query': result['prompt'],
            'response': filter_responses
        })
    except IndexError:
        print(result['prompt'])






print(count)
print(len(eval_funcs))
print(len(filter_samples))
filter_samples = list(map(json.loads, set(map(json.dumps, filter_samples))))
print(len(filter_samples))



with jsonlines.open("./output/eval_score_results.jsonl", "w") as f:
    for each in filter_samples:
        f.write(each)

