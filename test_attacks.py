import datetime
import random
import transformers
import json
import torch
import os, re
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import argparse
import torch.nn.functional as F
from transformers import AutoTokenizer
from attacks.generate_adv_samples import generate_one_adv_sample
from datasets_loader import BaseDataset
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, ResNetForImageClassification, ViTForImageClassification
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from utils.func import boolean_string
from attacks.helpers import *
from utils.metric import AverageMeter, accuracy, cosine_similarity


# Disable warning for tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

tqdm._instances.clear()

criterion = torch.nn.CrossEntropyLoss()

test_tranform = transforms.Compose([
                        transforms.Resize(size=(224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                    ])

def eval_model(args):
    start_time = datetime.datetime.now()
    disable_torch_init()
    print('==> Loading model ...')
    if  'clip' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        vision_model = CLIPVisionModelWithProjection.from_pretrained(args.model_name, output_hidden_states=True, torch_dtype=torch.float16).cuda().eval()
        text_model = CLIPTextModelWithProjection.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda().eval()
    elif 'resnet' in args.model_name.lower():
        vision_model = ResNetForImageClassification.from_pretrained(args.model_name, output_hidden_states=True, torch_dtype=torch.float16).cuda().eval()
    elif 'vit' in args.model_name.lower():
        vision_model = ViTForImageClassification.from_pretrained(args.model_name, output_hidden_states=True, torch_dtype=torch.float16).cuda().eval()

    print('==> Loading dataset ...')
    if args.dataset == 'imagenet':
        with open('/groups/sernam/datasets/imagenet/labels/imagenet_simple_labels.json') as f:
            label_list = json.load(f)
        dataset = ImageFolder(root="/datasets/ImageNet2012nonpub/validation", transform=test_tranform)
        dataset = Subset(dataset, random.sample(range(len(dataset)), args.subset))
        if 'clip' in args.model_name.lower():
            dataset.label_list = label_list
            text_labels = ["a photo of %s"%v for v in label_list]
            with torch.no_grad():
                text_labels = tokenizer(text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
                text_label_embeds = F.normalize(text_model(text_labels).text_embeds, p=2., dim=-1) # (N_labels, 768)
            dataset.text_label_embeds = text_label_embeds
        else:
            dataset.label_list = label_list
            dataset.text_label_embeds = None
    else:
        dataset = BaseDataset(args.dataset, model_name=args.model_name, path=None, subset=args.subset)
    
    dataloader = DataLoader(dataset, batch_size=16, num_workers=16, shuffle=False, pin_memory=True)
    acc1, acc5 = AverageMeter(), AverageMeter()
    for i, data in enumerate(tqdm(dataloader, position=0, leave=True)):

        if args.dataset == 'imagenet':
            images, labels = data
        else:
            images, _, labels = data 
        images = images.cuda().half()
        labels = labels.cuda()

        if args.attack_name is not None:

            if args.targeted:
                targeted_labels = torch.cat([get_different_class(c_true, dataset.label_list) for c_true in labels])
            else:
                targeted_labels = None

            images = generate_one_adv_sample(images, 
                args.attack_name, 
                args.model_name,
                vision_model, 
                text_label_embeds=dataset.text_label_embeds, 
                use_descriptors=False,
                use_last_n_hidden=args.use_last_n_hidden,
                save_image=args.save_image, 
                save_folder=None,
                save_names=None, 
                y=targeted_labels,
                eps=args.eps, 
                lr=args.lr, 
                nb_iter=args.nb_iter, 
                norm=args.norm,
                n_classes=len(dataset.label_list),
                initial_const=args.initial_const,
                binary_search_steps=args.binary_search_steps,
                confidence=args.confidence,)

        with torch.no_grad():

            if 'clip' in args.model_name:
                if (args.attack_name is not None) or (args.use_last_n_hidden is None) or (args.use_last_n_hidden == 1):
                    image_embeds = vision_model(images).image_embeds
                else:
                    image_embeds = vision_model(images).hidden_states[-args.use_last_n_hidden][:, 0, :]
                    image_embeds = list(list(vision_model.named_children())[0][1].named_children())[-1][1](image_embeds) # post layernorm
                    image_embeds = vision_model.visual_projection(image_embeds)       
        
                ## zero-shot result with image
                logits_per_image = cosine_similarity(text_embeds=dataset.text_label_embeds, image_embeds=image_embeds)
            elif 'vit' in args.model_name:
                if (args.attack_name is not None) or (args.use_last_n_hidden is None) or (args.use_last_n_hidden == 1):
                    logits_per_image = vision_model(images).logits
                else:
                    logits_per_image = vision_model(images).hidden_states[-args.use_last_n_hidden][:, 0, :]
                    logits_per_image = list(list(vision_model.named_children())[0][1].named_children())[-1][1](logits_per_image) # post layernorm
                    logits_per_image = vision_model.classifier(logits_per_image)
                    
            _acc1, _acc5 = accuracy(logits_per_image, labels, topk=(1, 5))
            acc1.update(_acc1[0].item())
            acc5.update(_acc5[0].item())

    end_time = datetime.datetime.now()
    print('Acc@1: {} Acc@5: {}'.format(acc1.avg, acc5.avg))
    with open(f"/groups/sernam/adv_llava/results/{args.dataset}_results.txt", "a+") as f:
        f.write('='*50+'\n')
        f.write(f"Job ID: {os.environ['SLURM_JOB_ID']}\n")
        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write('==> Acc@1: {} Acc@5: {}\n'.format(acc1.avg, acc5.avg))
        f.write("Start time: {} End time: {} Time elapsed: {}\n".format(start_time, end_time, end_time-start_time))
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--attack_name", type=str, default=None)
    parser.add_argument("--use_last_n_hidden", type=int, nargs='?' ,default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--save_image", type=boolean_string, default=False)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--grad_sparsity", type=int, default=99, help='grad sparsity for sparse_l1_descent')
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--norm", type=float, default=np.inf)
    parser.add_argument("--targeted", type=boolean_string, default=True)

    # Carlini Wagner attack
    parser.add_argument("--confidence", type=float, default=0)
    parser.add_argument("--initial_const", type=float, default=1e-2)
    parser.add_argument("--binary_search_steps", type=int, default=5)

    args = parser.parse_args()
    args.attack_name = None if args.attack_name == 'None' else args.attack_name

    # post-process args
    if args.attack_name is None or args.attack_name == 'None':
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
    
    eval_model(args)

