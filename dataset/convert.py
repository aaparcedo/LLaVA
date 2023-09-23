import json

def read_data_from_file(input_file_path):
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    return data

def write_data_to_file(output_file_path, data):
    with open(output_file_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

def remove_unwanted_strings(text, unwanted_strings):
    for unwanted_string in unwanted_strings:
        text = text.replace(unwanted_string, '')
    return text

def convert_format_1_to_format_2(input_file_path, output_file_path):
    # Load data from the input file
    data_format_1 = read_data_from_file(input_file_path)

    # Prepare an empty list for data in format 2
    data_format_2 = []
    question_id = 0

    # Iterate over each entry in the format 1 data
    for entry in data_format_1:
        # Iterate over each conversation in the entry
        for conversation in entry['conversations']:
            # Only consider 'human' responses
            if conversation['from'] == 'human':
                # Remove unwanted strings from the 'value' field
                conversation['value'] = remove_unwanted_strings(conversation['value'], ['<image>\n', '\n<image>'])
                # Create a new entry in format 2 and add it to the list
                entry_format_2 = {
                    'question_id': question_id,
                    'image': entry['image'],
                    'text': conversation['value'],
                    'category': 'conv'
                }
                data_format_2.append(entry_format_2)
                question_id += 1

    # Save the format 2 data to the output file
    write_data_to_file(output_file_path, data_format_2)

# Example usage
convert_format_1_to_format_2('conversation_58k.json', 'conversation_58k_questions.jsonl')
