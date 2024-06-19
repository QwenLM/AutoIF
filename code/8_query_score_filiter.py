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



results = list(jsonlines.open("./sample_data/query_rft_score.jsonl"))
filter_results = []
print(len(results))
for result in tqdm(results):
    scores = []
    for each in result['gen']:
        score = re.findall(r'Score: (\d+?)$', each)
        if score:
            scores.append(int(score[0]))
    score = np.mean(scores) if scores else 0
    if score > 8: # quality score
        filter_results.append(result)
print(len(filter_results))



with jsonlines.open("./output/query_score_filter.jsonl", "w") as f:
    for each in filter_results:
        f.write(each)
