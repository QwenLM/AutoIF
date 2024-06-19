import json

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
import tenacity

random.seed(0)



filter_results = []
count = 0 
filter_count=0
with open('./sample_data/back_trans.jsonl', 'r') as file:
    for i,line in enumerate(tqdm(file)):
        line = json.loads(line)
        back_instructions = line["back_instruction"]
        ori_ins = line['instruction']

        nli_scores=[]
        for back_ins in back_instructions[:3]:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            device = "cuda:0"
            model_name = "your path to mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
            premise = ori_ins
            hypothesis = back_ins

            input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
            output = model(input["input_ids"].to(device))  
            prediction = torch.softmax(output["logits"][0], -1).tolist()
            label_names = ["entailment", "neutral", "contradiction"]
            prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
            max_label = max(prediction, key=prediction.get)
            nli_scores.append(max_label)
      
            
            
        line["nli_scores"] = nli_scores
        print(nli_scores)

        if "contradiction" in nli_scores:
            filter_count+=1
            continue
        else:
            filter_results.append(line)
        
            
        count+=1


print("filter samples nums:",filter_count)

with jsonlines.open("./output/back_trans_fliter.jsonl", "w") as f:
    for each in filter_results:
        f.write(each)


