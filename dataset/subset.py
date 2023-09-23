import json
import random

def create_subset(input_file, output_file, percentage=0.01):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Calculate how many lines 1% would be
    subset_size = int(len(lines) * percentage)

    # Randomly sample lines
    subset = random.sample(lines, subset_size)

    # Write sampled lines to output file
    with open(output_file, 'w') as f:
        for line in subset:
            # Assuming each line is a valid JSON string
            json_obj = json.loads(line)
            json.dump(json_obj, f)
            f.write('\n')  # Write newline character after each JSON object

# Usage
create_subset('./subsets/adversarial_questions.jsonl', './subsets/adversarial_subset.jsonl', percentage=0.25)

