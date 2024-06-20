import json

data = []


with open('./sample_data/query_score_filter.jsonl', 'r', encoding='utf-8') as file:
    
    for dat in file:
        d = json.loads(dat)
        data.append(d)


processed_data = []
for item in data:
    
    item['query'] = item['query'][0].upper() + item['query'][1:]
    item['instruction'] = item['instruction'][0].upper()+ item['instruction'][1:]
    if "?" in item['query']:
        inputs = item['query']+" "+item['instruction']+"."

    elif "." in item['query']:
        inputs = item['query']+" "+item['instruction']+"."
    else:
        inputs=item['query']+". "+item['instruction']+"."


    new_item = {
        "instruction": inputs,
        "input": "",
        "output": item['response'],
        "history": []
    }


    processed_data.append(new_item)
print(len(processed_data))



#Save the SFT data as llama factory format

with open('./output/IF_sft_data.json', 'w', encoding='utf-8') as outfile:
    json.dump(processed_data, outfile, indent=4, ensure_ascii=False)
