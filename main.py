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
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from run_llava import run_LLaVA
from torch.utils.data import DataLoader, Subset
import datetime
from utils.metric import AverageMeter, accuracy
import random
# from functools import partial
# from tqdm import tqdm
# tqdm = partial(tqdm, position=0, leave=True)
from tqdm.auto import tqdm

# Disable warning for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"
denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)) # for denormalizing images

single_prompt="Please describe the image in one sentence."
descriptor_prompt="Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}"
llava_name = '/groups/sernam/ckpts/LLAMA-on-LLaVA'
clip_name = "openai/clip-vit-large-patch14"

disable_torch_init()

vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_name, torch_dtype=torch.float16).cuda().eval()
text_model = CLIPTextModelWithProjection.from_pretrained(clip_name, torch_dtype=torch.float16).cuda().eval()
clip_tokenizer = AutoTokenizer.from_pretrained(clip_name)

def get_dataloader(args):

    if args.save_image and args.adv_path:
        os.makedirs(args.adv_path, exist_ok=True)

    if args.dataset == 'imagenet':
        with open('/groups/sernam/datasets/imagenet/labels/imagenet_simple_labels.json') as f:
            label_list = json.load(f)
        dataset = ImageFolder(root="/datasets/ImageNet2012nonpub/validation", transform=transforms.Compose([
                        transforms.Resize(size=(args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                    ]))
        dataset = Subset(dataset, random.sample(range(len(dataset)), args.subset))
        dataset.label_list = label_list
        text_labels = ["a photo of %s"%v for v in label_list]
        with torch.no_grad():
            text_labels = clip_tokenizer(text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
            text_label_embeds = F.normalize(text_model(text_labels).text_embeds, p=2., dim=-1) # (N_labels, 768)
        dataset.text_label_embeds = text_label_embeds
    elif args.dataset == 'coco':
        dataset = BaseDataset(dataset=args.dataset, path=args.adv_path, task=args.task, image_size=args.image_size, subset=args.subset)
        
    dataloader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, pin_memory=True)
    return dataset, dataloader

def main(args):
    print(f"Slurm job ID: {os.environ.get('SLURM_JOB_ID', None)}")
    start_time = datetime.datetime.now()

    print('==> Start testing...')

    acc1, acc5 = AverageMeter(), AverageMeter()
    for data in tqdm(dataloader):
        if args.dataset == 'imagenet':
            images, labels = data
            base_paths = None
        else:
            images, base_paths, labels = data
        images = images.cuda().half()
        labels = labels.cuda()
        # before_response, before_image_cls_token = run_LLaVA(args, llava_model, llava_tokenizer, images)
        # transforms.ToPILImage()(denormalize(images[0])).save('./orig.png')
        # print('==> before: ', before_response)

        # args.adv_path != None means the images loaded are already adversarial
        if (not args.adv_path or args.adv_path != 'None') and (args.attack_name != 'None'):
            if args.targeted:
                targeted_labels = torch.cat([get_different_class(c_true, dataset.label_list) for c_true in labels])
            else:
                targeted_labels = None
            
            images = generate_one_adv_sample(images, 
                args.attack_name, 
                args.model_name,
                vision_model, 
                dataset.text_label_embeds, 
                use_descriptors=args.use_descriptors,
                save_image=args.save_image, 
                save_folder=args.adv_path,
                save_names=base_paths, 
                y=targeted_labels,
                eps=args.eps, 
                lr=args.lr, 
                nb_iter=args.nb_iter, 
                norm=args.norm,
                n_classes=len(dataset.label_list),
                initial_const=args.initial_const,
                binary_search_steps=args.binary_search_steps,
                confidence=args.confidence,)
            # transforms.ToPILImage()(denormalize(images[0])).save('./adv.png')

        with torch.no_grad():

            if 'llava' in args.model_name.lower():
                ## LLaVA text embedding
                text_response_embeds, image_cls_tokens = [], []
                for image in images:
                    image = image.unsqueeze(0)
                    response, image_cls_token = run_LLaVA(args, llava_model, llava_tokenizer, image)
                    text_response = clip_tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                    text_response_embed = F.normalize(text_model(text_response).text_embeds, p=2., dim=-1)
                    text_response_embeds.append(text_response_embed)
                    image_cls_tokens.append(image_cls_token)
                
                text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768
                logits = torch.matmul(text_response_embeds, dataset.text_label_embeds.t()).mean(dim=1)

            elif 'clip' in args.model_name.lower():
                image_embeds = F.normalize(vision_model(images).image_embeds, p=2., dim=-1)
                ## zero-shot result with clip
                logits = torch.matmul(image_embeds, dataset.text_label_embeds.t()) # B, n_label

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
    parser.add_argument("--model-name", type=str, default="['/groups/sernam/ckpts/LLAMA-on-LLaVA']")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "caption"])
    parser.add_argument("--dataset", type=str, default="['coco']")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')
    parser.add_argument("--adv_path", type=str, required=False, default=None)

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
    parser.add_argument("--generate_one_response", type=boolean_string, default='False')
    parser.add_argument("--query", type=str, default="[descriptor_prompt]")

    
    args = parser.parse_args()

    queries = eval(args.query)
    eps = eval(args.eps)
    model_names = eval(args.model_name)
    datasets = eval(args.dataset)
    for model_name in model_names:
    # replace with automodel?
        torch.cuda.empty_cache()
        args.model_name = model_name
        if 'llava' in model_name.lower():
            model_name = os.path.expanduser(model_name)
            llava_tokenizer = AutoTokenizer.from_pretrained(model_name)
            llava_model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()

        for ds in datasets:
            args.dataset = ds
            dataset, dataloader = get_dataloader(args)
            for query in queries:
                args.query = query
                for ep in eps:
                    args.eps = ep

                    print("==> Running with args: dataset: {} model_name: {} eps: {} query: {}".format(ds, model_name, ep, query))

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

                    main(args)

                    print("=" * 50)
