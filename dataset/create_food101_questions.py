import os
import json

def create_question_list(root_folder, output_file):
    question_id = 0

    # Open the output file in write mode
    with open(output_file, 'w') as outfile:

        # Iterate over each subdirectory in the root folder
        for class_name in os.listdir(root_folder):
            class_folder = os.path.join(root_folder, class_name)

            # Check that it's actually a directory (and not a file or a link)
            if os.path.isdir(class_folder):

                # Iterate over each file in the class folder
                for image_name in os.listdir(class_folder):
                    image_path = os.path.join(class_folder, image_name)

                    # Check that it's a file (and not a directory or a link)
                    if os.path.isfile(image_path):

                        # Create a new dictionary for this question
                        question_dict = {
                            "question_id": question_id,
                            "image": image_name,
                            "object_name": class_name.replace("_", " ").capitalize(),
                            "class_id": class_name,  # Assuming class_id is the same as the class_name
                            "text": "Can you fill in the blank, This is a photo of a []",
                            "category": "detail"
                        }

                        # Write the question dictionary to the file as a JSON object
                        json.dump(question_dict, outfile)

                        # Write a newline to separate JSON objects
                        outfile.write('\n')

                        # Increment the question_id for the next question
                        question_id += 1

# Use the function
root_folder = "/home/crcvreu.student2/food-101/images"
output_file = "/home/crcvreu.student2/LLaVA/dataset/subsets/food101_questions.jsonl"
create_question_list(root_folder, output_file)

