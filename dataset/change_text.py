import json

# Open the input and output files
with open('./full_dataset/food101_questions.jsonl', 'r') as infile, open('./subsets/food101_questions.jsonl', 'w') as outfile:
    # Iterate over each line in the input file
    for line in infile:
        # Load the line as a JSON object
        obj = json.loads(line)

        # Change the "text" value to "hello world"
        obj["text"] = "Can you fill in the blank, This is a photo of a []"

        # Write the modified object to the output file
        outfile.write(json.dumps(obj) + '\n')
