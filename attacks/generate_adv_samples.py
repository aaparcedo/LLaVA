from tqdm import tqdm
import transformers
import json
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
from utils.func import make_descriptor_sentence
from llava.utils import disable_torch_init
from llava.model import *
import torchvision.transforms as transforms
from .helpers import * 
import torch.nn.functional as F
from .pgd import pgd
from .sparse_l1 import sl1d
from .cw import cw

os.environ["TOKENIZERS_PARALLELISM"] = "true"

denormalize = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), 
                                   (1/0.229, 1/0.224, 1/0.225)) # for denormalizing images

# def generate_one_adv_sample(image, 
#                             attack_name, 
#                             model_name,
#                             vision_model,
#                             text_label_embeds=None, 
#                             use_descriptors=False,
#                             y=None,
#                             save_image=True, 
#                             save_folder=None, 
#                             save_names=None,
#                             use_last_n_hidden=None,
#                             lr=0.01,
#                             nb_iter=30,
#                             **kwargs) -> torch.Tensor:

#     if save_image and not (save_folder or save_names or save_folder == 'None'):
#         raise ValueError('If save_image is True, save_path must be provided.')
#     if text_label_embeds is not None:
#         text_label_embeds.requires_grad = True

#     attack = eval(attack_name)
#     adv_image = attack(
#         model_fn=lambda x: compute_output_logits(x, vision_model, text_label_embeds, use_descriptors=use_descriptors, 
#                                                  use_last_n_hidden=use_last_n_hidden, model_name=model_name), 
#         x=image,
#         y=y,
#         targeted=(y is not None),
#         lr=lr,
#         nb_iter=nb_iter,
#         **kwargs
#     )

#     if save_image: 
#         denormalized_tensor = denormalize(adv_image)
#         for i in range(len(save_names)):
#             save_image = denormalized_tensor[i]
#             save_image = T.ToPILImage()(save_image)
#             save_image.save(os.path.join(save_folder,save_names[i]))
    
#     return adv_image

def generate_one_adv_sample(args,
                            image, 
                            model,
                            text_label_embeds=None,
                            save_names=None,
                            y=None,
                            n_classes=None) -> torch.Tensor:

    if args.save_image and not (args.save_folder or save_names or args.save_folder == 'None'):
        raise ValueError('If save_image is True, save_path must be provided.')
    if text_label_embeds is not None:
        text_label_embeds.requires_grad = True
    attack = eval(args.attack_name)
    adv_image = attack(
        model_fn=lambda x: model(x, text_label_embeds, is_attack=True), 
        x=image,
        y=y,
        targeted=(y is not None),
        lr=args.lr,
        eps=args.eps, 
        nb_iter=args.nb_iter,
        initial_const=args.initial_const,
        norm=args.norm,
        binary_search_steps=args.binary_search_steps,
        confidence=args.confidence,
        grad_sparsity=args.grad_sparsity,
        n_classes=n_classes
    )

    if args.save_image: 
        denormalized_tensor = denormalize(adv_image)
        for i in range(len(save_names)):
            save_image = denormalized_tensor[i]
            save_image = T.ToPILImage()(save_image)
            save_image.save(os.path.join(args.save_folder,os.path.basename(save_names[i])))
    
    return adv_image