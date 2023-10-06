import transformers
import json
import torch
import os, re
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import argparse
from transformers import AutoTokenizer
from datasets import BaseDataset
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
Vanilla
"""



mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1,3,1,1))
std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1,3,1,1))

criterion = torch.nn.CrossEntropyLoss()

def get_different_class(c_true, classes):
    classes_kept = [c for c in classes if c != c_true]
    return np.random.choice(classes_kept)

def eval_model(args):

    if args.dataset == 'coco':
        image_list = '/groups/sernam/datasets/coco/coco_test.txt'
        label_list = '/groups/sernam/datasets/coco/coco_test.json'
        # args.subset = 50
    
    elif args.dataset == 'imagenet':
        image_list = '/groups/sernam/datasets/imagenet/imagenet_test.txt'
        label_list = '/groups/sernam/datasets/imagenet/imagenet_label.json'
        # args.subset = 5000
    
    else:
        raise NotImplementedError("Dataset not implemented")

    f = open(image_list, 'r')
    image_list = f.readlines()
    f.close()

    label_names = set()
    for line in image_list:
        label = line.split('|')[-1].split('\n')[0]
        label_names.add(label)
    label_names = sorted(list(label_names))



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

    num_image_correct = 0

    image_list = image_list[:50]
    label_all = []

    for v in label_names:
        label_all.append("a photo of %s"%v)
    with torch.no_grad():
        text_labels = tokenizer(label_all, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = text_model(text_labels).text_embeds
        text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

    dataset = BaseDataset( path=None, subset=50)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, pin_memory=True)

    for i, data in enumerate(tqdm(dataloader)):
        # image_name, label_name = data.split('|')
        # image = Image.open(image_name).convert('RGB')
        # image = test_transform(image).cuda().half().unsqueeze(0)

        # ## Text label embedding
        # label_name = label_name.split('\n')[0]
        # label = label_names.index(label_name)
        image, _, label = data 
        image = image.cuda().half()

        with torch.no_grad():

            image_embeds = vision_model(image).image_embeds
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    
            ## zero-shot result with image
            logits_per_image = torch.matmul(image_embeds, text_label_embeds.t())
            
            if logits_per_image.argmax(dim=-1).item()==label:
                num_image_correct += 1

    print(num_image_correct/len(image_list))

    # print(i + args.set*args.set_num, "Image: %.4f"%(num_image_correct/(i+1)), "Text: %.4f"%(num_text_correct/(i+1)), "adv Acc: %.4f"%(num_adv_correct/(i+1)), "adv Text Acc: %.4f"%(num_adv_text_correct/(i+1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./LLAMA-on-LLaVA")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--image_size", type=float, default=224)

    args = parser.parse_args()

    eval_model(args)

