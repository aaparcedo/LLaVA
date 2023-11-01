import timm
from PIL import Image
from tqdm import tqdm
import os
import json
import argparse
from torchattacks import PGD
from utils.atk_utils import get_pred
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoProcessor
from typing import Any, List, Union
import numpy as np
import torchattacks
from attacks.torchattacks_wrapper import apgd


# print('vit_large_patch16_224_in21k')
# print('resnet18')


# For encoding labels/captions and the generated response, we use the same CLIP text encoder
clip_text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
# clip_text_encoder = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()
clip_text_encoder = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32).cuda().eval()

clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')

class CLIP(nn.Module):
    def __init__(self, args, label_names):
        super().__init__()
        self.args = args
        # self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()
        # self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32).cuda().eval()
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=torch.float16).cuda()

        # self.processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')

        self.label_list = label_names
            
        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
        self.text_label_embeds = self._encode_labels()
        

    def forward(self, images, text_label_embeds=None, is_attack=False, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """
        # image_embeds = self.vision_encoder(images).image_embeds
        image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
                ## zero-shot result with clip
        logits = torch.matmul(image_embeds, self.text_label_embeds.t()) # B, n_label
        return logits
    
    @torch.no_grad()
    def _encode_labels(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        return a tensor of shape (N_labels, 768). This is for encoding all the classification class labels,
        so we don't have to recompute for every image.
        """
        print("==> Loading text label embeddings...")
        text_labels = ["a photo of %s" % v for v in self.label_list]
        text_labels = [label.split('\n')[0] for label in text_labels]  # Remove newline characters
        text_labels = self.processor(text=text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = F.normalize(self.text_encoder(text_labels).text_embeds, p=2., dim=-1)
        return text_label_embeds

def run_classification(args):
    
    # Load label list
    f = open(args.label_list, 'r')
    
    unique_label_names = set()
    
    for line in f:
        line = json.loads(line)
        unique_label_names.add(line['text'])
        
    label_names = list(unique_label_names)
        
    f.close()

    # Load image list
    f = open(args.image_list, 'r')
    image_list = f.readlines()
    f.close()
        
    model = CLIP(args, label_names)
    
    print(model)
        
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
     
    
    atk = apgd(model)
    atk.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    
    print(atk)

    num_correct = 0
    num_adv_correct = 0
    
    # Subset image list
    image_list = image_list[:1000]
    
    for i, data in tqdm(enumerate(image_list)):
        image_name, label_name = data.split('|') # label_name: 'cat'
        image = Image.open(image_name).convert('RGB')
        base_name = os.path.basename(image_name)
        
        label_name = label_name.split('\n')[0]
        label_index = label_names.index(label_name)

        image_tensor = transform(image).unsqueeze(0).cuda()
        label_tensor = torch.tensor([int(label_index)], dtype=torch.long).cuda()
        
        adv_image = atk(image_tensor, label_tensor)
        
        pred = get_pred(model, image_tensor, 'cuda')
        adv_pred = get_pred(model, adv_image, 'cuda')
        
        if label_index == pred:
            num_correct += 1
        
        if label_index == adv_pred:
            num_adv_correct += 1
            
        
    print(f'CLIP Accuracy (no attack): {(num_correct / len(image_list)) * 100}%')
    print(f'CLIP Accuracy after attack: {(num_adv_correct / len(image_list)) * 100}%')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_list", type=str, default="/groups/sernam/datasets/coco/coco_trainval2014.jsonl")
    parser.add_argument("--image_list", type=str, default="/groups/sernam/datasets/coco/coco_test.txt")  
    parser.add_argument("--model", type=str, default='clip')
    parser.add_argument("--attack", type=str, default='apgd')
    args = parser.parse_args()
    
    log_file = f'/home/aaparcedo/LLaVA/attacks/results/{args.attack}.log' # TODO: change this

    # sys.stdout = open(log_file, 'w')
    
    run_classification(args)


