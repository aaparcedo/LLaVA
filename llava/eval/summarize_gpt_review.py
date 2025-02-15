import json
import os
from collections import defaultdict

import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-d', '--dir')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    review_files = [x for x in os.listdir(args.dir) if x.endswith('.json') and (x.startswith('gpt4_text') or x.startswith('reviews_'))]

    print(f'review file: {review_files}')
    for review_file in sorted(review_files):
        config = review_file.replace('gpt4_text_', '').replace('.jsonl', '')
        scores = defaultdict(list)
        print(f'GPT-4 vs. {config}')
        with open(os.path.join(args.dir, review_file)) as f:
            for review_str in f:
                review = json.loads(review_str)
                scores[review['category']].append(review['tuple'])
                scores['all'].append(review['tuple'])
        for k, v in scores.items():
            print(f'iteration: {k}, score: {len(v)}')
           
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            print(f'stats[0]: {stats[0]}, stats[1]: {stats[1]}')
            print(k, stats, round(stats[1]/stats[0]*100, 1))
        print('=================================')
