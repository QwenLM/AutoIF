




import json
import random
import itertools



input_file_path = './sample_data/dpo_query_eval_score_results.jsonl'


output_file_path = './output/dpo_pairs.jsonl'


def random_sample(lst, n):
    """Randomly sample n items from a list without replacement.
    If the list does not contain enough items, return the whole list."""
    return random.sample(lst, min(n, len(lst)))

# Open and read the input JSON file line by line
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        # Load the JSON data from the current line
        data = json.loads(line)

        instruction =data['query']

        # Filter and separate the positive and negative cases
        positive_cases = [case[0] for case in data["response"] if case[1] >= 0.5 ]
        negative_cases = [case[0] for case in data["response"] if case[1] == 0]

        # Check if there are enough cases for sampling
        if len(positive_cases) >= 2 or len(negative_cases) >= 2:
            # Skip the line if there are not enough positive or negative cases


            # Randomly select 2 positive and 2 negative samples
            positive_samples = random_sample(positive_cases, 2)
            negative_samples = random_sample(negative_cases, 2)
        elif len(positive_cases) >=1  or len(negative_cases) >= 1:
            # Skip the line if there are not enough positive or negative cases


            # Randomly select 2 positive and 2 negative samples
            positive_samples = random_sample(positive_cases, 1)
            negative_samples = random_sample(negative_cases, 1)
        else:
            continue
            # Skip the line if there are not enough positive or negative cases


            # Randomly select 2 positive and 2 negative samples
            positive_samples = random_sample(positive_cases, 2)
            negative_samples = random_sample(negative_cases, 2)

        # Generate all possible combinations of these samples (4 combinations)
        combinations = list(itertools.product(positive_samples, negative_samples))
        # import pdb
        # pdb.set_trace()
        combinations = set(combinations)


        # For each combination, append a new entry to 'new_json_data'
        for comb in combinations:
            # Create the new JSON object for the current combination
            new_json_data = json.dumps({
                "instruction": instruction,
                "positive": comb[0],
                "negative": comb[1]
            })
            # Write the new JSON line to the output file
            outfile.write(new_json_data + '\n')  # Add newline character after each JSON line

print("The output JSON with all possible combinations has been successfully created at '{}'.".format(output_file_path))


'''
Tips:
The dpo data construction is similar in online and offline DPO, the main difference is the supervision model for responses.


Please TODO
Need to convert eval_score_results into Llama factory data format
'''