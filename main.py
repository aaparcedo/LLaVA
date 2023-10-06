from typing import Tuple
import json
import torch
import os
import numpy as np
from PIL import Image
import argparse
from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from attacks.generate_adv_samples import generate_one_adv_sample
from datasets import BaseDataset
from utils.func import make_descriptor_sentence
from llava.utils import disable_torch_init
from llava.model import *
import torchvision.transforms as transforms
from attacks.helpers import *

import torch.nn.functional as F
from run_llava import run_LLaVA
from torch.utils.data import DataLoader
import datetime
from utils.metric import AverageMeter, accuracy
# from functools import partial
# from tqdm import tqdm
# tqdm = partial(tqdm, position=0, leave=True)
from tqdm.auto import tqdm

# Disable warning for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main(args):

    start_time = datetime.datetime.now()

    disable_torch_init()

    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    model_name = os.path.expanduser(args.model_name)
    llava_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llava_model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()

    vision_model.eval()
    text_model.eval()

    if args.attack_name != 'None': 

        adv_img_folder = f"/groups/sernam/adv_llava/adv_datasets/{args.dataset}/{args.task}_{args.attack_name}_lr{args.lr}_steps{args.nb_iter}"
        if args.attack_name == 'pgd':
            adv_img_folder += f"_eps{args.eps}_norm{args.norm}"
        elif args.attack_name == 'sparse_l1_descent':
            adv_img_folder += f"_eps{args.eps}_grad_sparsity{args.grad_sparsity}"
        else:
            adv_img_folder += f"_bss{args.binary_search_steps}"
        # check if the corresponding adv image folder already exist
        # adv_path != None means the target adverarial dataset is already generated
        if os.path.exists(adv_img_folder):
            adv_path = adv_img_folder
        else:
            if args.save_image:
                os.makedirs(adv_img_folder)
            adv_path=None
    else:
        adv_path=None


    dataset = BaseDataset(dataset=args.dataset, path=adv_path, task=args.task, image_size=args.image_size, subset=args.subset)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    print('==> Start testing...')

    clip_acc1, clip_acc5, llava_acc1, llava_acc5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for images, base_paths, labels in tqdm(dataloader):
        images = images.cuda().half()
        labels = labels.cuda()

        if (not adv_path) and (args.attack_name != 'None'):
            images = generate_one_adv_sample(images, 
                args.attack_name, 
                dataset.text_label_embeds, 
                vision_model, 
                use_descriptors=args.use_descriptors,
                save_image=args.save_image, 
                save_folder=adv_img_folder,
                save_names=base_paths, 
                eps=args.eps, 
                lr=args.lr, 
                nb_iter=args.nb_iter, 
                norm=args.norm)
            
        with torch.no_grad():
            ## LLaVA text embedding
            text_response_embeds, image_cls_tokens = [], []
            for image in images:
                image = image.unsqueeze(0)
                response, image_cls_token = run_LLaVA(args, llava_model, llava_tokenizer, image)
                text_response = tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_response_embed = F.normalize(text_model(text_response).text_embeds, p=2., dim=-1)
                text_response_embeds.append(text_response_embed)
                image_cls_tokens.append(image_cls_token)
            
            image_cls_tokens = torch.cat(image_cls_tokens, dim=0) 
            text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768
            image_embeds = F.normalize(vision_model.visual_projection(image_cls_tokens), p=2., dim=-1)

            ## zero-shot result with clip
            logits_per_image = torch.matmul(image_embeds, dataset.text_label_embeds.t()) # B, n_label

            ## zero-shot result with llava response
            logits_per_text = torch.matmul(text_response_embeds, dataset.text_label_embeds.t()).mean(dim=1) # B, n_label
        
            _clip_acc1, _clip_acc5 = accuracy(logits_per_image, labels, topk=(1, 5))
            _llava_acc1, _llava_acc5 = accuracy(logits_per_text, labels, topk=(1, 5))
            clip_acc1.update(_clip_acc1[0].item())
            clip_acc5.update(_clip_acc5[0].item())
            llava_acc1.update(_llava_acc1[0].item())
            llava_acc5.update(_llava_acc5[0].item())

    end_time = datetime.datetime.now()
    print("** CLIP Acc@1: {}  LlaVa Acc@1: {} **".format(clip_acc1.avg, llava_acc1.avg))
    print("** CLIP Acc@5: {}  LlaVa Acc@5: {} **".format(clip_acc5.avg, llava_acc5.avg))
    time_elapsed = end_time - start_time
    with open(f"/groups/sernam/adv_llava/results/{args.dataset}_results.txt", "a+") as f:
        f.write('='*50+'\n')
        f.write(f"Job ID: {os.environ['SLURM_JOB_ID']}\n")
        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("** CLIP Acc@1: {}  LlaVa Acc@1: {} ** \n".format(clip_acc1.avg, llava_acc1.avg))
        f.write("** CLIP Acc@5: {}  LlaVa Acc@5: {} ** \n".format(clip_acc5.avg, llava_acc5.avg))
        f.write("Start time: {} End time: {} Time elapsed: {}\n".format(start_time, end_time, time_elapsed))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/groups/sernam/ckpts/LLAMA-on-LLaVA")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "caption"])
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')

    # args for adv attack
    parser.add_argument("--attack_name", type=str, default="pgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--grad_sparsity", type=int, default=99, help='grad sparsity for sparse_l1_descent')
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--norm", type=float, default=np.inf)
    parser.add_argument("--binary_search_steps", type=int, default=5)
    
    parser.add_argument("--use_descriptors", type=boolean_string, default='False')

    parser.add_argument("--query", type=str, default="Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}")

    
    args = parser.parse_args()

    # post-process args
    if args.attack_name == 'None':
        args.lr = None
        args.nb_iter = None
        args.norm = None
        args.eps = None
        args.grad_sparsity = None
        args.binary_search_steps = None
    
    if args.attack_name == 'pgd':
        args.grad_sparsity = None
        args.binary_search_steps = None
    elif args.attack_name == 'sl1d':
        args.grad_sparsity = None
        args.norm = None
        args.binary_search_steps = None
    elif args.attack_name == 'cw2':
        args.grad_sparsity = None
        args.eps = None
        args.norm = None

    for k,v in vars(args).items():
        print(f"{k}: {v}")
    main(args)