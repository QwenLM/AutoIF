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



def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

results = list(jsonlines.open("./sample_data/query_rft.jsonl"))

filter_samples = []
for result in tqdm(results):
    eval_funcs = []


    for func, score in result['eval_func']:
        local_vars = {}
        exec(func, globals(), local_vars)
        eval_funcs.append(local_vars['evaluate'])
    
    filter_responses = []

    for response in result['gpt-answer']:
        acc = []
        for eval_func in eval_funcs:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
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


        if acc > 0:
            filter_responses.append(response)

    for each in filter_responses:
        try:
            filter_samples.append({
                'instruction': result['instruction'],
                'query': re.findall(r'\[Query\](.*)$', result['prompt'], re.DOTALL)[0].strip(),
                'response': each
            })
        except IndexError:
            print(result['prompt'])


print(len(eval_funcs))
print(len(filter_samples))
filter_samples = list(map(json.loads, set(map(json.dumps, filter_samples))))
print(len(filter_samples))


# Save the data with out score consistency
with jsonlines.open("./output/query_wo_score.jsonl", "w") as f:
    for each in filter_samples:
        f.write(each)

prompt_template = """You are an expert that is good at judging whether a response is following the instruction and query.
[Instruction] {instruction}
[Query] {query}
[Response] {response}
Please notice that the response may not be helpful as it needs to strictly follow the requirements in the Instruction.
You need to judge whether the response answers the query. Please first provide a detailed analysis and then give a score ranking from 0 to 10 at the last line.
Scoring 0 means the response is totally unrelated to the query, while scoring 10 means the response is helpful and highly related to the query.
Please only provide a score in the format `Score: {{score}}` without any other contents at the last line."""

for each in filter_samples:
    each['prompt'] = prompt_template.format(
        instruction=each['instruction'],
        query=each['query'],
        response=each['response']
    )

# Save the data with out scoring prompt
with jsonlines.open("./output/query_need_quality_score.jsonl", "w") as f:
    for each in filter_samples:
        f.write(each)


'''
Please TODO:

Please score the consistency for each query and response
'''