# Apply some transformation to image

import argparse
import json
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import albumentations as A

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def apply_transformation(x, augmentation):
    """
    Apply some data augmentation technique to an image. 
    :param x: Image tensor.
    :param augmentation: Augmentation technique to be applied to image x.
    :return: Transformed image tensor.
    """
    
    # ROTATE 90 DEGREES
    if augmentation == "rot_90":
        rotated_image = TF.rotate(img=x, angle=90) 
        # rotated_image.save(f'./pics/{path}/{image_id}_rot_90.jpg')
        return rotated_image
    
    # ROTATE 180 DEGREES
    if augmentation == "rot_180":
        rotated_image = TF.rotate(img=x, angle=180) 
        # rotated_image.save(f'./pics/{path}/{image_id}_rot_180.jpg')
        return rotated_image
    
    # ROTATE 270 DEGREES
    if augmentation == "rot_270":
        rotated_image = TF.rotate(img=x, angle=270) 
        # rotated_image.save(f'./pics/{path}/{image_id}_rot_270.jpg')
        return rotated_image

    # CROP AND RESIZE THE IMAGE
    if augmentation == "crop_resize":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            T.Resize(size=(224, 224)),
            T.RandomCrop((112, 112)),  # Perform a random crop
            T.Resize((224, 224)),  # Resize the cropped image back to the original size
        ])
        crop_resize_image = transform(x)

        return crop_resize_image
    
    # CROP, RESIZE, AND FLIP THE IMAGE
    if augmentation == "crop_resize_flip":
        original_size = x.size 
        height, width = original_size
        crop_width = int(original_size[0] * 0.5)
        crop_height = int(original_size[1] * 0.5)

        # Define the transformations
        transform = T.Compose([
            T.RandomCrop((crop_height, crop_width)),  # Perform a random crop
            T.Resize((width, height)),  # Resize the cropped image back to the original size
            T.RandomHorizontalFlip(p=1.0)
            ])

        crop_resized_flipped_image = transform(x)
        # crop_resized_flipped_image.save(f'./pics/{path}/{image_id}_crop_resize_flip.jpg')
        return crop_resized_flipped_image
    
    # TODO: check coco_cutout.py to make this work.
    # In other file we divide cutout_image by 255 and then normalize for some reason but it works
    if augmentation == 'cutout':
        transform = A.Compose([
            A.Resize(224, 224),
            A.CoarseDropout(max_holes=1, max_height=112, max_width=112, p=1),
        ])
        cutout_image = torch.Tensor((transform(image=np.array(x))["image"])).half().permute(2, 0, 1)
        return cutout_image
    
    # COLOR JITTER THE IMAGE
    if augmentation == "color_jitter":
        # Add a random number generator for these pararameters
        transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5)
        color_jitter_image = transform(x)
        # color_jitter_image.save(f'./pics/{path}/{image_id}_color_jitter.jpg')
        return color_jitter_image
    
    # GRAYSCALE THE IMAGE
    if augmentation == "grayscale":
        transform = T.Grayscale()
        grayscale_image = transform(x)
        # grayscale_image.save(f'/home/crcvreu.student2/LLaVA/subset_1%_experiments/experiment_v3/detail/images/{path}/{image_file}')
        return grayscale_image
    
    # GAUSSIAN NOISE
    # if augmentation == "gaussian_noise":
        
    #     # Define the noise level
    #     noise_level = 0.1
    #     # Convert PIL image to tensor
    #     tensor_image = T.ToTensor()(x)

    #     # Generate Gaussian noise
    #     noise = torch.randn_like(tensor_image) * noise_level

    #     # Add the Gaussian noise to the image
    #     noisy_image_tensor = tensor_image + noise

    #     # Convert tensor to PIL image
    #     noisy_image = T.ToPILImage()(noisy_image_tensor)


    #     # Save the noisy image as a .jpg file
    #     noisy_image.save(f'/home/crcvreu.student2/LLaVA/subset_1%_experiments/experiment_v3/detail/images/{path}/{image_file}')

    #     return noisy_image

    # GAUSSIAN BLUR
    if augmentation == "gaussian_blur":
        transform = T.GaussianBlur(17, sigma=3)
        gaussian_blur_image = transform(x)
        return gaussian_blur_image

    # SOBEL FILTER
    # if augmentation == "sobel_filter":
    #     grayscale_transform = T.Grayscale()
    #     grayscale_image = grayscale_transform(x)

    #     grayscale_image = T.ToTensor()(grayscale_image)

    #     img_tensor = grayscale_image.unsqueeze(0)

    #     # Define Sobel kernels
    #     sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    #     sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])

    #     # Add extra dimensions to the kernels for batch and channel
    #     sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3)
    #     sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3)

    #     # Move the kernels to the same device as the image
    #     sobel_kernel_x = sobel_kernel_x.to(img_tensor.device)
    #     sobel_kernel_y = sobel_kernel_y.to(img_tensor.device)

    #     # Apply the Sobel kernels to the image
    #     edge_x = F.conv2d(img_tensor, sobel_kernel_x, padding=1)
    #     edge_y = F.conv2d(img_tensor, sobel_kernel_y, padding=1)

    #     # Combine the x and y edge images
    #     edge = torch.sqrt(edge_x**2 + edge_y**2)

    #     # Remove the extra batch dimension
    #     edge = edge.squeeze(0)

    #     # Convert the tensor to an image
    #     edge_image = T.ToPILImage()(edge)

    #     # Save the edge-detected image
    #     # edge_image.save(f'./pics/{path}/{image_id}_sobel_filter.jpg')

    #     return edge_image


def apply_transformation_to_dir(args):

    denormalize = T.Normalize(mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711], std=[1/0.26862954, 1/0.26130258, 1/0.27577711])

    # Resize (for CLIP), turn image to tensor, and normalize the tensor
    preprocess_transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

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

        image = preprocess_transform(image)

        # Apply transformation
        transformed_image = apply_transformation(x=image, augmentation=args.augmentation_technique)

        # Denormalize back to "normal" values

        # 
        if args.normalize == True:
            print("in normalize")
            
        transformed_image = denormalize(transformed_image).half()

        # In case we want to save the rgb pil image
        if args.save_image:
            pil_image = T.ToPILImage()(transformed_image)
            os.makedirs(f"/home/crcvreu.student2/coco/{args.augmentation_technique}_rgb", exist_ok=True)
            pil_image.save(f'/home/crcvreu.student2/coco/{args.augmentation_technique}_rgb/{base_name}.png') #

        # Add batch dimension to the tensor we'll save
        transformed_image = transformed_image.unsqueeze(0)

        # Save transformed image as .pt
        tmp = image_name.split(os.sep)
        tmp[tmp.index('trainval2014')] = f'{args.augmentation_technique}'
        os.makedirs(f"/home/crcvreu.student2/coco/{args.augmentation_technique}", exist_ok=True)
        save_name = os.path.splitext(os.sep + os.path.join(*tmp))[0] + ".pt"
        torch.save(transformed_image, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--set", type=int, required=True)
    parser.add_argument("--set_num", type=int, required=True)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False') # If True save pil image as well
    parser.add_argument("--augmentation_technique", type=str, required=True, help="Enter augmentation to apply")
    parser.add_argument("--normalize", type=boolean_string, default='False', required=False, help="Set to True to normalize")


    args = parser.parse_args()

    apply_transformation_to_dir(args)
