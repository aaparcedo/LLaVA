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

def generate_one_adv_sample(args,
                            image, 
                            model,
                            text_label_embeds=None,
                            text=None,
                            save_names=None,
                            y=None,
                            save_format='pt',
                            n_classes=None) -> torch.Tensor:

    if text_label_embeds is not None:
        text_label_embeds.requires_grad = True


    attack = eval(args.attack_name)
    adv_image = attack(
        model_fn=lambda x: model(x, text_label_embeds=text_label_embeds, text=text, is_attack=True), 
        x=image,
        y=y,
        targeted=args.targeted,
        use_ce_loss=getattr(args, "use_ce_loss", False),
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
        detached_adv_image = adv_image.detach().cpu()
        denormalized_tensor = denormalize(detached_adv_image)
        for i in range(len(save_names)):
            save_image_path = os.path.join(args.save_folder,save_names[i])
            os.makedirs(os.path.split(save_image_path)[0], exist_ok=True)

            if save_format == 'pt':
                torch.save(detached_adv_image[i], os.path.join(args.save_folder, os.path.splitext(save_names[i])[0]+'.pt'))
            else:
                save_image = denormalized_tensor[i]
                save_image = T.ToPILImage()(save_image)
                save_image.save(save_image_path)
    
    return adv_image


