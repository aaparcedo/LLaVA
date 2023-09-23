import json
import random

# Read the data from the JSONL file
with open('./line_format/instruction_large_questions.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Get unique question ids
question_ids = list(set([item['question_id'] for item in data]))

# Calculate 1% of total question ids
one_percent_count = max(1, len(question_ids) // 100)

# Select 1% of question ids randomly
selected_question_ids = random.sample(question_ids, one_percent_count)

# Select lines that belong to the selected question ids
subset_data = [item for item in data if item['question_id'] in selected_question_ids]

# Write the subset to a new JSONL file
with open('./subsets/instruction_large_1%_subset.jsonl', 'w') as f:
    for item in subset_data:
        f.write(json.dumps(item))
        f.write('\n')
