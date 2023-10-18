import json
import shortuuid
from datasets_loader import VQAV2LLaVAGenerationDataset
from torch.utils.data import DataLoader
from run_llava import run_LLaVA
from tqdm import tqdm
from vqav2.vqa import VQA
from vqav2.vqaEval import VQAEval
import argparse
from PIL import Image
import os
import numpy as np
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from llava.model import *
import torch
from m4c_evaluator import EvalAIAnswerProcessor


def generate_responses(args):

    llava_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llava_model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda().eval()

    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    with open(args.questions_file, 'r') as f:
        questions = np.array([json.loads(l) for l in f.readlines()])
    
    if args.subset is not None:
        questions = np.random.permutation(questions)[:args.subset]

    answer_processor = EvalAIAnswerProcessor()

    for question in tqdm(questions):
        question_id = question['question_id']
        image_name = question['image']
        question = question['text']
        image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")
        image = processor(images=image, return_tensors="pt").pixel_values.cuda().half()

        outputs, _ = run_LLaVA(args, llava_model, llava_tokenizer, image, question, vqa=True)
        with open(args.output_file, 'a') as f:
            ans_id = shortuuid.uuid()
            f.write(json.dumps({"question_id": question_id,
                                "prompt": question,
                                "text": answer_processor(outputs),
                                "answer_id": ans_id}) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/groups/sernam/ckpts/LLAMA-on-LLaVA')
    parser.add_argument('--questions_file', type=str, default='/groups/sernam/datasets/vqa/vqav2/coco2014val_questions_llava.jsonl')
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1_1')
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    generate_responses(args)
