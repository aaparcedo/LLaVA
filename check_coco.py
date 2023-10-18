import json
import os
from tqdm import tqdm

id_len = 12
with open('/groups/sernam/datasets/vqa/vqav2/v2_mscoco_val2014_annotations.json', 'r') as f:
    annotations = json.load(f)['annotations']
    ids = []
    for anno in annotations:
        ids.append(str(anno['image_id']))
    
    ids = list(set(ids))

matches = 0
with open('/groups/sernam/datasets/coco/coco_2014val.txt', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, total=len(lines), position=0, leave=True):
        if 'COCO_train2014' not in line:
            path, label = line.split('|')
            path = os.path.basename(path)
            for id in ids:
                if id in line:
                    matches += 1
            print(matches)
                


print(matches)


