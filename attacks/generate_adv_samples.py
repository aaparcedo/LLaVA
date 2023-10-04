from tqdm import tqdm
import transformers
import json
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
from utils.func import compute_img_text_logits, make_descriptor_sentence
from llava.utils import disable_torch_init
from llava.model import *
import torchvision.transforms as transforms
from .helpers import * 
import torch.nn.functional as F
from .pgd import pgd
from .sparse_l1 import sl1d
from .cw import cw2

denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), 
                                   (1/0.229, 1/0.224, 1/0.225)) # for denormalizing images

def generate_one_adv_sample(image, 
                            attack_name, 
                            text_label_embeds, 
                            vision_model, 
                            use_descriptors=False,
                            save_image=True, 
                            save_path=None, 
                            lr=0.01,
                            nb_iter=30,
                            **kwargs) -> torch.Tensor:

    if save_image and not save_path:
        raise ValueError('If save_image is True, save_path must be provided.')
    attack = eval(attack_name)
    adv_image = attack(
        model_fn=lambda x: compute_img_text_logits(x, text_label_embeds, vision_model, use_descriptors=use_descriptors), 
        x=image,
        lr=lr,
        nb_iter=nb_iter,
        **kwargs
    )

    if save_image: 
        denormalized_tensor = denormalize(adv_image)
        save_image = denormalized_tensor.squeeze(0)
        save_image = T.ToPILImage()(save_image)
        save_image.save(save_path)
    
    return adv_image

def generate_adversarials_pgd(args):

    if args.dataset == 'coco':
        image_list = './coco_test.txt'
        label_list = './coco_test.json'
        descriptor_list = './descriptors/descriptors_coco.json'
        # args.subset = 50
    
    elif args.dataset == 'imagenet':
        image_list = './imagenet_test.txt'
        label_list = './imagenet_label.json'
        descriptor_list = './descriptors/descriptors_imagenet.json'
        # args.subset = 5000
    
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    label_all = []

    if args.use_descriptors:
        with open(descriptor_list, 'r') as f:
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
    

    disable_torch_init()

    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
     
    vision_model.eval()
    text_model.eval()

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
        text_label_embeds = text_model(text_labels).text_embeds # (B, N_labels, 768)
        text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

    if args.save_image:
        os.makedirs(f"../datasets/{args.dataset}/{args.attack_name}_eps{args.eps}_lr{args.lr}_steps{args.nb_iter}_norm{args.norm}", exist_ok=True)

    for i, data in tqdm( enumerate(image_list)):
        image_name, label_name = data.split('|')
        base_name = os.path.basename(image_name)
        image = Image.open(image_name).convert('RGB')
        image = test_transform(image).cuda().half().unsqueeze(0)

        ## Text label embedding
        label_name = label_name.split('\n')[0]
        label_name = "a photo of " + label_name

        generate_one_adv_sample(image, 
                            args.attack_name, 
                            text_label_embeds, 
                            vision_model, 
                            use_descriptors=args.use_descriptors,
                            save_image=args.save_image, 
                            image_name=base_name, 
                            lr=args.lr,
                            nb_iter=args.nb_iter,
                            y=get_different_class(label_name, label_all) if args.targeted else None,
                            targeted=args.targeted,
                            norm=args.norm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_name", type=str, default="zebra.jpeg")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--attack_name", type=str, default="pgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--norm", type=float, default=np.inf)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')
    parser.add_argument("--descriptors", type=boolean_string, required=False, default='False')

    args = parser.parse_args()
    args.dataset = args.image_list.split('/')[-1].split('_')[0]
    generate_adversarials_pgd(args)