import timm
from PIL import Image
from tqdm import tqdm
import os
import json
import argparse
from torchattacks import PGD
from utils import get_pred
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoProcessor
from typing import Any, List, Union
import numpy as np
import torchattacks


""" for debugger, to save tensor as image to check how imperceptible attack is

denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)) 
test1 = denormalize(adv_image).squeeze(0).permute(1, 2, 0).cpu().numpy()
test1 = (test1 * 255).astype(np.uint8)
test1 = Image.fromarray(test1)
test1.save('/home/aaparcedo/adversarial-attacks-pytorch/demo/first_image.png')


"""

# print('vit_large_patch16_224_in21k')
# print('resnet18')


# For encoding labels/captions and the generated response, we use the same CLIP text encoder
clip_text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
# clip_text_encoder = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()
clip_text_encoder = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32).cuda().eval()
clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')

class CLIP(nn.Module):
    def __init__(self, args, label_list):
        super().__init__()
        self.args = args
        # self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float32).cuda().eval()
        self.processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
        
        self.label_list = label_list
        
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
    

        
    # transform = timm.data.create_transform(
    #     **timm.data.resolve_data_config(model.pretrained_cfg)
    # )

    # Load label list
    f = open(args.label_list, 'r')
    # label_names = list(json.load(f).keys())
    
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
        
    if args.model == 'resnet':
        model = timm.create_model('resnet18', pretrained=True).eval().cuda()
    elif args.model == 'vit':
        model = timm.create_model('vit_large_patch16_224_in21k', pretrained=True).eval().cuda()
    elif args.model == 'clip':
        model = CLIP(args, label_names)
        
        
        
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
     
    
    # atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    # atk = torchattacks.CW(model, c=1, kappa=1, steps=50, lr=0.01)
    # atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=80, seed=None, verbose=False)
    # atk = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
    # atk = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
    # atk = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
    # atk = torchattacks.EADEN(model, kappa=0, lr=0.01, max_iterations=100)
    # atk = torchattacks.EADL1(model, kappa=0, lr=0.01, max_iterations=100)
    # atk = torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)
    atk.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    
    print(atk)


    num_correct = 0
    num_adv_correct = 0
    
    # Subset image list
    # image_list = [image_list[i] for i in np.random.choice(len(image_list), 100, replace=False)]
    
    image_list = image_list[:100]
    
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
    print(f'len of image list: {len(image_list)}')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--label_list", type=str, default="/groups/sernam/datasets/imagenet/labels/imagenet_label.json")
    # parser.add_argument("--image_list", type=str, default="/groups/sernam/datasets/imagenet/labels/imagenet_test.txt")   
    parser.add_argument("--label_list", type=str, default="/groups/sernam/datasets/coco/coco_trainval2014.jsonl")
    parser.add_argument("--image_list", type=str, default="/groups/sernam/datasets/coco/coco_test.txt")  
    parser.add_argument("--model", type=str, default='clip')
    args = parser.parse_args()
    
    log_file = f'/home/aaparcedo/LLaVA/attacks/results/deepfool_coco.log' 

    sys.stdout = open(log_file, 'w')
    
    run_classification(args)

