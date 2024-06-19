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




filter_results=[]

with jsonlines.open("./sample_data/back_trans.jsonl", "r") as f:
    for each in f:
        filter_results.append(each)



sft_data = list(jsonlines.open("your path to sharegot dataset"))
queries = [each['messages'][1]['content'] for each in sft_data if each['source'] == 'en:sharegpt']

queries = [each for each in queries if len(each) > 20 and len(each) < 300]

inputs = []
for instruction in tqdm(filter_results):
    ins_queries = random.sample(queries, 16) #拼16个
    for q in ins_queries:
        prompt = f"Please answer the query strictly following the instruction.\n[instruction] {instruction['instruction']}\n[Query] {q}"
        item = copy.deepcopy(instruction)
        item['prompt'] = prompt
        inputs.append(item)
        # import pdb
        # pdb.set_trace()

with jsonlines.open("./output/instruction_filtered_llama3_72b_query_sampled.jsonl", "w") as f:
    for each in inputs:
        f.write(each)


'''
Please TODO:

Please use supervision model perform RFT to generate k Responses for each query
'''