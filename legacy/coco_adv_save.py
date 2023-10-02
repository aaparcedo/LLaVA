import transformers
import json
import torch
import os, re
import numpy as np
from PIL import Image
from torchvision.utils import save_image

import argparse
from transformers import AutoTokenizer
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
import torchvision.transforms as transforms

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1,3,1,1))
std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1,3,1,1))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

criterion = torch.nn.CrossEntropyLoss()

def get_different_class(c_true, classes):
    classes_kept = [c for c in classes if c != c_true]
    return np.random.choice(classes_kept)

def adv_sample_generation(vision_model, image, text_embeds, adv_label, temp=0.05, LR = 0.5, steps=50):

    """
        Young's implementation of PGD
    """
    
    for step in range(steps):
        image.requires_grad = True

        image_embeds = vision_model(image).image_embeds
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())

        loss = criterion(logits_per_image / temp, torch.Tensor([adv_label]).reshape([1]).cuda().long())
        # print(step, logits_per_image.argmax(dim=-1).item(), adv_label, loss.item())

        image.retain_grad()
        loss.retain_grad()
        loss.backward(retain_graph=True)

        image_grad = image.grad.detach().cpu().numpy()
        image_np = image.detach().cpu().numpy()- LR*image_grad
        image = torch.Tensor(image_np).cuda().half()

    return image


def eval_model(args):

    f = open(args.label_list, 'r')
    label_names = list(json.load(f).keys())
    f.close()

    label_all = []

    for v in label_names:
        label_all.append("a photo of %s"%v)

    f = open(args.image_list, 'r')
    image_list = f.readlines()
    f.close()

    disable_torch_init()

    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    test_transform = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
     
    vision_model.eval()
    text_model.eval()

    image_list = image_list[args.set*args.set_num:(args.set+1)*args.set_num]
    print(len(image_list))

    with torch.no_grad():

        text_labels = tokenizer(label_all, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = text_model(text_labels).text_embeds
        text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

    for i, data in enumerate(image_list):
        image_name, label_name = data.split('|')
        image = Image.open(image_name).convert('RGB')
        image = test_transform(image).cuda().half().unsqueeze(0)

        ## Text label embedding
        label_name = label_name.split('\n')[0]


        adv_label = get_different_class(label_name, label_names)
        adv_label = label_names.index(adv_label)
        adv_image = adv_sample_generation(vision_model, image, text_label_embeds, adv_label)

        print(f'\n\nimage name: {image_name}')
        print(f'shape adv image: {adv_image.shape}')
        print(adv_image)

        
        tmp = image_name.split(os.sep)
        print(tmp)
        tmp[tmp.index('trainval2014')] = 'adv_trainval2014'
        save_name = os.path.splitext(os.sep + os.path.join(*tmp))[0] + ".pt"
        torch.save(adv_image, save_name)
        print(i + args.set*args.set_num, save_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./LLAMA-on-LLaVA")
    parser.add_argument("--image_list", type=str, default="./coco_test.txt")
    parser.add_argument("--label_list", type=str, default="./coco_label.json")
    parser.add_argument("--set", type=int, required=True)
    parser.add_argument("--set_num", type=int, required=True)
    parser.add_argument("--image_size", type=float, default=224)
    parser.add_argument("--llava_temp", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--adv", type=boolean_string, default='True')
    parser.add_argument("--query", type=str, default="Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}")
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
