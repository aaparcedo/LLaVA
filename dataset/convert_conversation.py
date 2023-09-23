import json

def convert_format(data):
    converted_data = []
    convo_count = 0  # Initialize conversation count
    for conversation in data:
        question_count = 0
        for i in range(0, len(conversation['conversations']), 2):
            if conversation['conversations'][i]['from'] == 'human' and conversation['conversations'][i+1]['from'] == 'gpt':
                new_data = {}
                new_data['question_id'] = convo_count
                new_data[f'conv_{convo_count}_question_id'] = question_count
                new_data['image'] = conversation['image']

                # Replace unwanted strings in conversation text
                human_text = conversation['conversations'][i]['value'].replace('\n<image>', '').replace('<image>\n', '')
                
                new_data['text'] =  human_text # remove the last comma and space
                new_data['category'] = 'conv'
                converted_data.append(new_data)
                question_count += 1
        convo_count += 1  # Increment conversation count after each conversation
    return converted_data

# Load data from JSON file
with open('./full_dataset/conversation_58k.json', 'r') as f:
    data = json.load(f)

converted_data = convert_format(data)

# Write the results to a new JSONL file
with open('./line_format/conversation_58k_questions.jsonl', 'w') as f:
    for item in converted_data:
        f.write(json.dumps(item))
        f.write('\n')
