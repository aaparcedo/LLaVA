# This python script can be used to convert the conversation types "instruction_questions" dataset
# The "intruction_questions" dataset has all the conv types as "conv" because of the convertion file
# The point of this is to change the conv type to "complex" if the text appears in the complex dataset
# And to change the conv type to "detial" if the text appears in the detail dataset

import json

# Load the data from the JSONL files
with open('./line_format/instruction_small_questions.jsonl', 'r') as f:
    combined = [json.loads(line) for line in f]
with open('./line_format/detail_23k_questions.jsonl', 'r') as f:
    detail = [json.loads(line) for line in f]
with open('./line_format/complex_reasoning_77k_questions.jsonl', 'r') as f:
    complex = [json.loads(line) for line in f]

# Create sets of the 'text' values for faster lookup
detail_texts = set(item['text'] for item in detail)
complex_texts = set(item['text'] for item in complex)

# Initialize counters for 'detail' and 'complex' changes
detail_changes = 0
complex_changes = 0

# Iterate over the items in the combined data
for item in combined:
    # If the 'text' value is found in the detail data, change the 'category' to 'detail'
    if item['text'] in detail_texts:
        item['category'] = 'detail'
        detail_changes += 1
    # If the 'text' value is found in the complex data, change the 'category' to 'complex'
    elif item['text'] in complex_texts:
        item['category'] = 'complex'
        complex_changes += 1

# Write the updated data back to the combined JSONL file
with open('./line_format/combined_small.jsonl', 'w') as f:
    for item in combined:
        f.write(json.dumps(item) + '\n')

# Print the total number of 'detail' and 'complex' changes
print(f'Total detail changes: {detail_changes}')
print(f'Total complex changes: {complex_changes}')

