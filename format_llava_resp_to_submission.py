import os
import argparse
import json
from tqdm import tqdm
from m4c_evaluator import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    answer_processor = EvalAIAnswerProcessor()
    results = []
    error_line = 0
    for line_idx, line in tqdm(enumerate(open(args.src))):
        try:
            line = json.loads(line) 
            results.append({"question_id": line['question_id'], "answer": answer_processor(line['text'])})
        except:
            error_line += 1

    print(f'total results: {len(results)}, error_line: {error_line}')
    

    # for x in test_split:
    #     if x['question_id'] not in results:
    #         all_answers.append({
    #             'question_id': x['question_id'],
    #             'answer': ''
    #         })
    #     else:
    #         all_answers.append({
    #             'question_id': x['question_id'],
    #             'answer': answer_processor(results[x['question_id']])
    #         })

    with open(args.dst, 'w') as f:
        json.dump(results, f)
