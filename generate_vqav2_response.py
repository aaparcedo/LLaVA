import json
import shortuuid
from datasets_loader import get_dataloader
from torch.utils.data import DataLoader
from run_llava import run_LLaVA
from tqdm import tqdm
from utils.func import boolean_string
from vqav2.vqa import VQA
from vqav2.vqaEval import VQAEval
import argparse
from PIL import Image
import os
import numpy as np
from models import get_model
import torch
from m4c_evaluator import EvalAIAnswerProcessor
from utils.helpers import get_model_name_from_path

def generate_responses(args):
    
    model = get_model(args)

    _, dataloader = get_dataloader(args, model)

    answer_processor = EvalAIAnswerProcessor()
    paths = args.image_folder.split('/')
    model_name = get_model_name_from_path(args.model_path)
    answers_file = os.path.join("/groups/sernam/adv_llava/results/responses/vqav2/", 
                                paths[-2], 
                                "{}_{}_{}.json".format(model_name, paths[-1], os.environ.get('SLURM_JOB_ID', '')))
    results = []
    for inputs_ids, image_tensor, question_id in tqdm(dataloader):
        output = model.generate(inputs_ids, image_tensor)
        results.append({"question_id": question_id[0].item(), "answer": answer_processor(output)})
    with open(answers_file, 'a') as f:
        json.dump(results, f)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/groups/sernam/ckpts/LLAMA-on-LLaVA')
    parser.add_argument('--data-file', type=str, default='/groups/sernam/datasets/vqa/vqav2/coco2014val_questions.jsonl')
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--dataset", type=str, default='coco')
    parser.add_argument("--task", type=str, default='vqa')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_format", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    # llava params
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1_1')
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)

    # for some models (e.g. BLIP2), if we want single word response, we have to restrict the number of new tokens
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_len", type=int, default=1)

    args = parser.parse_args()

    for k, v in vars(args).items():
        print("{}: {}".format(k, v))

    generate_responses(args)
