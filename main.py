from typing import Tuple
import json
import torch
import os
import numpy as np
from PIL import Image
import argparse
from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from attacks.generate_adv_samples import generate_one_adv_sample
from func import make_descriptor_sentence
from llava.utils import disable_torch_init
from llava.model import *
import torchvision.transforms as transforms
from attacks.helpers import *
from tqdm import tqdm
import torch.nn.functional as F
from run_llava import run_LLaVA

@torch.inference_mode()
def get_text_image_pred(args, image_embeds, text_label_embeds, text_response_embeds, label) -> Tuple[bool, bool]:

    attention_scores = torch.nn.functional.softmax(torch.mm(image_embeds, text_response_embeds.t()) / args.temp, dim=1)
    weighted_sum = torch.mm(attention_scores, text_response_embeds)
    combined_embeds = F.normalize(image_embeds * (1.0-args.scale) + weighted_sum * args.scale, p=2., dim=-1)

    ## zero-shot result with image
    logits_per_image = torch.matmul(image_embeds, text_label_embeds.t())

    ## zero-shot result with llava response
    # logits_per_text = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=0)
    logits_per_text = torch.matmul(combined_embeds, text_label_embeds.t())
    
    return logits_per_image.argmax(dim=-1).item()==label, \
            logits_per_text.argmax(dim=-1).item()==label


@torch.inference_mode()
def main(args):

    if args.dataset == 'coco':
        image_list = './coco_test.txt'
        label_list = './coco_test.json'
        # args.subset = 50
    
    elif args.dataset == 'imagenet':
        image_list = './imagenet_test.txt'
        label_list = './imagenet_label.json'
        # args.subset = 5000
    
    else:
        raise NotImplementedError("Dataset not implemented")

    with open(image_list, 'r') as f:
        image_list = f.readlines()

    if args.use_descriptors:
        with open(label_list, 'r') as f:
            descriptors = json.load(f)
            label_names = list(descriptors.keys())
    else:
        # ==> Sam: label_list format incorrect, a hot fix
        label_names = set()
        for line in image_list:
            label = line.split('|')[-1].split('\n')[0]
            label_names.add(label)
        label_names = list(label_names)
        # ==> end of hot fix

        # f = open(label_list, 'r')
        # label_names = list(json.load(f).keys())
        # f.close()

    image_list = np.random.permutation(image_list)[:args.subset]

    disable_torch_init()

    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    test_transform = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    vision_model.eval()
    text_model.eval()

    num_image_correct = 0
    num_text_correct = 0

    if args.use_descriptors:
        text_label_embeds = []

        for label in label_names:
            examples = descriptors[label]
            sentences = []
            for example in examples:
                sentence = f"{label} {make_descriptor_sentence(example)}"
                sentences.append(sentence)
                
            text_descriptor = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_descriptor_embeds = F.normalize(text_model(text_descriptor).text_embeds, p=2., dim=-1) # (B, N_descriptions, 768)
            text_label_embeds.append(text_descriptor_embeds) # list of (B, N_descriptions, 768) len = # N_labels
    else:
        text_labels = ["a photo of %s"%v for v in label_names]
        text_labels = tokenizer(text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = F.normalize(text_model(text_labels).text_embeds, p=2., dim=-1) # (B, N_labels, 768)

    # check if the corresponding adv image folder already exist
    adv_img_folder = f"../datasets/{args.dataset}/{args.attack_name}_eps{args.eps}_lr{args.lr}_steps{args.nb_iter}_norm{args.norm}"
    if os.path.exists():
        adv_exist = True
    else:
        adv_exist = False

    for i, path in enumerate(tqdm(image_list)):
        # load & transform image
        image_path, label_name = path.split('|')
        if not args.adv:
            image = Image.open(image_path).convert('RGB')
            image = test_transform(image).cuda().half().unsqueeze(0)
        else:
            # if the corresponding adv images are already generated, load them
            # else, gererate them on the fly
            if adv_exist:
                image = Image.open(os.path.join(adv_img_folder, os.path.basename(image_path))).convert('RGB')
                image = test_transform(image).cuda().half().unsqueeze(0)
            else:
                image = Image.open(image_path).convert('RGB')
                image = test_transform(image).cuda().half().unsqueeze(0)
                image = generate_one_adv_sample(image, 
                                args.attack_name, 
                                text_label_embeds, 
                                vision_model, 
                                use_descriptors=args.use_descriptors,
                                save_image=args.save_image, 
                                image_name=os.path.basename(image_path), 
                                eps=args.eps, 
                                lr=args.lr, 
                                nb_iter=args.nb_iter, 
                                norm=args.norm)
        
        ## Text label embedding
        label_name = label_name.split('\n')[0]
        label = label_names.index(label_name)

        ## LLaVA text embedding
        response, image_cls_token = run_LLaVA(args, image)
        image_embeds = F.normalize(vision_model.visual_projection(image_cls_token), p=2., dim=-1)
        
        text_response = tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
        text_response_embeds = F.normalize(text_model(text_response).text_embeds, p=2., dim=-1)

        is_image_correct, is_text_correct = get_text_image_pred(args, image_embeds, text_label_embeds, text_response_embeds, label)
        
        if is_image_correct:
            num_image_correct += 1 # CLIP

        if is_text_correct:
            num_text_correct += 1 # LLaVA

    with open("./outputs/results.txt", "a") as f:
        f.write('='*50+'\n')
        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("==> Image Acc: {}  Text Acc: {}".format(num_image_correct/len(image_list), num_text_correct/len(image_list)))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./ckpts/LLAMA-on-LLaVA")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--image_size", type=float, default=224)
    parser.add_argument("--attack_name", type=str, default="pgd")
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--adv", type=boolean_string, default='True')
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--norm", type=float, default=np.inf)
    parser.add_argument("--query", type=str, default="Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--use_descriptors", type=boolean_string, default='False')
    args = parser.parse_args()

    # post-process args
    if not args.adv:
        args.lr = None
        args.nb_iter = None
        args.norm = None

    main(args)