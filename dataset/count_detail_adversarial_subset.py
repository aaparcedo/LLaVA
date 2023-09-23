# Count the matches of adversarial image examples that appear in the 1% detail subset

import json

# Function to read jsonl file
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

# Load the first file
data1 = read_jsonl('./subsets/detail_1%_subset.jsonl')

# Load the second file
data2 = read_jsonl('./subsets/adversarial_questions.jsonl')

# Extract image IDs from both files
image_ids1 = {item['image'] for item in data1}
image_ids2 = {item['image'] for item in data2}

# Find the intersection of the two sets
matching_ids = image_ids1 & image_ids2

# Print the count of matching IDs
print(len(matching_ids))

