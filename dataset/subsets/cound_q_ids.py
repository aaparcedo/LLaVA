import json

def count_unique_ids(filename):
    unique_ids = set()
    
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            unique_ids.add(data['question_id'])
            
    return len(unique_ids)

# usage
filename = 'instruction_large_1%_subset.jsonl'  # replace with your filename
count = count_unique_ids(filename)
print(f'Number of unique question ids: {count}')

