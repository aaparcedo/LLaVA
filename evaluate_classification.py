from typing import Tuple
import json
import torch
import os
import numpy as np
from PIL import Image
import argparse
from attacks.generate_adv_samples import generate_one_adv_sample
from datasets_loader import CLIPDataset, LLAVA2Dataset
from llava.utils import disable_torch_init
from llava.model import *
import torchvision.transforms as transforms
from attacks.helpers import *
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import datetime
from utils.metric import AverageMeter, accuracy
import random
# from functools import partial
# from tqdm import tqdm
# tqdm = partial(tqdm, position=0, leave=True)
from tqdm.auto import tqdm
from models import get_model
from datasets_loader import get_dataloader

# Disable warning for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"
one_word_response = "What is the main object in this image?\nAnswer the question using a single word or phrase."
disable_torch_init()

def main(model, args):
    print(f"Slurm job ID: {os.environ.get('SLURM_JOB_ID', None)}")
    start_time = datetime.datetime.now()

    print('==> Start testing...')

    acc1, acc5 = AverageMeter(), AverageMeter()
    for data in tqdm(dataloader):

        images, base_paths, labels = data
        images = images.cuda().half()
        labels = labels.cuda()

        with torch.no_grad():
            
            if 'clip' in args.model_path:
                logits = model(images, text_label_embeds=dataset.text_label_embeds)
            else:
                logits = model(images, text_label_embeds=dataset.text_label_embeds, input_ids=dataset.query_input_ids.unsqueeze(0))
            _acc1, _acc5 = accuracy(logits, labels, topk=(1, 5))

            acc1.update(_acc1[0].item())
            acc5.update(_acc5[0].item())

    end_time = datetime.datetime.now()
    print("** Acc@1: {} Acc@5: {} **".format(acc1.avg, acc5.avg))
    time_elapsed = end_time - start_time
    with open(f"/groups/sernam/adv_llava/results/{args.task}_{args.dataset}_results.txt", "a+") as f:
        f.write('='*50+'\n')
        f.write(f"Job ID: {os.environ['SLURM_JOB_ID']}\n")
        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("** Acc@1: {} Acc@5: {} **".format(acc1.avg, acc5.avg))
        f.write("Start time: {} End time: {} Time elapsed: {}\n".format(start_time, end_time, time_elapsed))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="['openai/clip-vit-large-patch14']")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "retrieval"])
    parser.add_argument("--dataset", type=str, default="['coco']")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1_1') #### vicuna_v1_1 can generate one-word response.
    parser.add_argument("--subset", type=int, default=None, nargs='?', help="number of images to test")
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--image-folder", type=str, required=False, default=None)
    parser.add_argument("--annotation-file", type=str, required=False, default=None)
    parser.add_argument("--query", type=str, default="[one_word_response]")

    args = parser.parse_args()

    queries = eval(args.query)
    model_paths = eval(args.model_path)
    datasets = eval(args.dataset)

            
    for model_path in model_paths:
        torch.cuda.empty_cache()
        args.model_path = model_path
        model = get_model(args)

        for ds in datasets:
            args.dataset = ds
            dataset, dataloader = get_dataloader(args, model)
            
            for query in queries:
                args.query = query

                print("==> Evaluating: dataset: {} model_path: {} query: {}".format(ds, model_path, query))

                for k,v in vars(args).items():
                    print(f"{k}: {v}")

                main(model, args)

                print("=" * 50)
