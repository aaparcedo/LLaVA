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

if __name__ == '__main__':
    select_subset('/groups/sernam/datasets/imagenet_val2012.jsonl', '/groups/sernam/datasets/imagenet_val2012_subset1000.jsonl')