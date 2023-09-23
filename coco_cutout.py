import argparse
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
from tqdm import tqdm
import torchvision.transforms as T
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


def apply_cutout(args):

    transforms_cutout = A.Compose([
    A.Resize(224, 224),
    A.CoarseDropout(max_holes=1, max_height=112, max_width=112, p=1),
    ToTensorV2(),
    ])

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


    if args.dataset == 'coco':
        image_list = './coco_test.txt'
        label_list = './coco_label.json'
    elif args.dataset == 'imagenet':
        image_list = './imagenet_test.txt'
        label_list = './imagenet_label.json'
    else:
        print("Wrong dataset!")
        return

    with open(image_list, 'r') as f:
        image_list = f.readlines()

    with open(label_list, 'r') as f:
        label_names = list(json.load(f).keys())

    image_list = image_list[args.set*args.set_num:(args.set+1)*args.set_num]

    for i, data in tqdm(enumerate(image_list), total=len(image_list)):
        image_name, label_name = data.strip().split('|')
        base_name = os.path.splitext(os.path.basename(image_name))[0]  # Get the name without extension
        image = Image.open(image_name).convert('RGB')

        np_img = np.array(image)

        transformed = transforms_cutout(image=np_img)
        transformed_image = transformed["image"]
        # transformed_image = transformed_image.float()

        cutout_pil_image = T.ToPILImage()(transformed_image) # Convert cutout tensor to PIL

        image_np = transformed_image.cpu().numpy() # Converting our cutout tensor to np array
        tensor_to_save = torch.Tensor(image_np).cuda().half() # Our tensor after converting to NP and back to tensor

        
        tensor_to_save = tensor_to_save / 255.0 # Set values between [0, 1]
        tensor_to_save = normalize(tensor_to_save) # Normalize
        tensor_to_save = tensor_to_save.unsqueeze(0) # Add batch dimension

        os.makedirs("/home/crcvreu.student2/coco/cutout_png", exist_ok=True)
        cutout_pil_image.save(f'/home/crcvreu.student2/coco/cutout_png/{base_name}.png') # Save cutout pil image
        # Save the transformed (cutout) images as tensors for later use in training
        tmp = image_name.split(os.sep)

        tmp[tmp.index('trainval2014')] = 'cutout'
        save_name = os.path.splitext(os.sep + os.path.join(*tmp))[0] + ".pt"
        torch.save(tensor_to_save, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--set", type=int, required=True)
    parser.add_argument("--set_num", type=int, required=True)
    args = parser.parse_args()

    apply_cutout(args)
