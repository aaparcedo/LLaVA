from typing import Tuple
import json
import torch
import os
import numpy as np
from PIL import Image
import argparse
from attacks.generate_adv_samples import generate_one_adv_sample
from datasets_loader import CLIPDataset, LLAVA2Dataset, get_dataloader
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

# Disable warning for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"
denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)) # for denormalizing images

single_prompt="Please describe the image in one sentence."
descriptor_prompt="Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}"

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

        if (not args.adv_path or args.adv_path == 'None') and (args.attack_name != 'None' or args.attack_name is not None):
            if args.targeted:
                targeted_labels = torch.cat([get_different_class(c_true, dataset.label_list) for c_true in labels])
            else:
                targeted_labels = None

            images = generate_one_adv_sample(
                args,
                images, 
                model,
                dataset.descriptors_embeds if args.attack_descriptors else dataset.text_label_embeds, 
                save_names=base_paths, 
                y=targeted_labels,
                n_classes=len(dataset.label_list))

        with torch.no_grad():

            logits = model(images, dataset.descriptors_embeds if args.use_descriptors else dataset.text_label_embeds)
            _acc1, _acc5 = accuracy(logits, labels, topk=(1, 5))

            # if _acc1 < 100:
            #     print(f'==> Misclassified image: {base_paths[0]}, label: {labels[0]}')
            #     print('Before: ', before_response)
            #     print('After: ', response)
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
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "caption"])
    parser.add_argument("--dataset", type=str, default="['coco']")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False', help='Whether to save generated adversarial images')
    parser.add_argument("--image_folder", type=str, required=False, default=None)

    # args for adv attack
    parser.add_argument("--attack_name", type=str, default="pgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--eps", type=str, default="[0.2]")

    # sparse l1 descent
    parser.add_argument("--grad_sparsity", type=int, default=99, help='grad sparsity for sparse_l1_descent')

    parser.add_argument("--norm", type=float, default=np.inf)

    parser.add_argument("--use_last_n_hidden", type=int, default=1)
    parser.add_argument("--targeted", type=boolean_string, default='False')

    # Carlini Wagner attack
    parser.add_argument("--binary_search_steps", type=int, default=5)
    parser.add_argument("--initial_const", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0)

    parser.add_argument("--use_descriptors", type=boolean_string, default='False')
    parser.add_argument("--attack_descriptors", type=boolean_string, default='False')
    parser.add_argument("--generate_one_response", type=boolean_string, default='False')
    parser.add_argument("--query", type=str, default="[descriptor_prompt]")

    args = parser.parse_args()

    queries = eval(args.query)
    eps = eval(args.eps)
    model_paths = eval(args.model_path)
    datasets = eval(args.dataset)

    if args.attack_descriptors and not args.use_descriptors:
        raise ValueError("If attack_descriptors is True, use_descriptors must be True")
            
    for model_path in model_paths:
        torch.cuda.empty_cache()
        args.model_path = model_path
        model = get_model(args)

        for ds in datasets:
            args.dataset = ds
            dataset, dataloader = get_dataloader(args, model)
            
            if ds != 'imagenet' and args.use_descriptors:
                raise ValueError("Only imagenet dataset has descriptors")
            
            for query in queries:
                args.query = query
                for ep in eps:
                    args.eps = ep

                    print("==> Running with args: dataset: {} model_path: {} eps: {} query: {}".format(ds, model_path, ep, query))

                    # post-process args
                    if args.attack_name == 'None':
                        args.lr = None
                        args.nb_iter = None
                        args.norm = None
                        args.eps = None
                        args.grad_sparsity = None
                        args.binary_search_steps = None
                        args.confidence = None
                        args.initial_const = None
                        args.targeted = None
                    
                    elif args.attack_name == 'pgd':
                        args.grad_sparsity = None
                        args.binary_search_steps = None
                        args.confidence = None
                        args.initial_const = None
                    elif args.attack_name == 'sl1d':
                        args.grad_sparsity = None
                        args.norm = None
                        args.binary_search_steps = None
                        args.confidence = None
                        args.initial_const = None
                    elif args.attack_name == 'cw':
                        args.grad_sparsity = None
                        args.eps = None
                        args.norm = None
                    else:
                        raise NotImplementedError(f"Attack {args.attack_name} not implemented")

                    for k,v in vars(args).items():
                        print(f"{k}: {v}")

                    main(model, args)

                    print("=" * 50)
