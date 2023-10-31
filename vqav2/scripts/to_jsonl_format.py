import json
from tqdm import tqdm
import os
import numpy as np

def file_formatter(path, output_path):
    with open(path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        for line in tqdm(lines):
            image, text = line.split('|')
            image, text = image.strip(), text.strip()
            image = os.path.basename(image)
            json.dump({"image": image, "text": text}, f)
            f.write('\n')


def select_subset(path, output_path=None, subset=1000, random=True):
    if path.endswith('.jsonl') or path.endswith('.txt'):
        with open(path, 'r') as f:
            lines = f.readlines()

        subset = lines[:subset] if not random else np.random.permutation(lines)[:subset]
        with open(output_path, 'w') as f:
            for line in tqdm(subset):
                f.write(line)

def select_subset_by_datafile(path, path_to_select, output_path=None):
    with open(path_to_select, 'r') as f:
        lines = f.readlines()
        images = set([json.loads(line)['image'] for line in lines])
    if path.endswith('.jsonl') or path.endswith('.txt'):

        with open(path, 'r') as f:
            lines = f.readlines()

        with open(output_path, 'w') as f:
            for line in tqdm(lines):
                if json.loads(line)['image'] in images:
                    f.write(line)
    else:
        with open(path, 'r') as f:
            data = json.load(f)
        questions = data['questions']
        filtered = []
        with open(output_path, 'a') as f:
            for q in tqdm(questions):
                image = "COCO_val2014_" + str(q['image_id']).zfill(12) + '.jpg'
                if image in images:
                    filtered.append(q)
            results = data
            results['questions'] = filtered
            json.dump(results, f)

if __name__ == '__main__':
    select_subset_by_datafile('/groups/sernam/datasets/vqa/vqav2/v2_OpenEnded_mscoco_val2014_questions.json', 
                              '/groups/sernam/datasets/coco/coco_val2014_subset1000.jsonl',
                              '/groups/sernam/datasets/vqa/vqav2/v2_OpenEnded_mscoco_val2014_questions_subset1000.json')