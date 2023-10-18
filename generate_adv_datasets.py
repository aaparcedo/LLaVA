
from tqdm import tqdm
import transformers
import json
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
from datasets_loader import BaseDataset
from models import get_model
from utils.metric import AverageMeter, accuracy
from utils.func import make_descriptor_sentence
from llava.utils import disable_torch_init
from attacks.helpers import * 
from attacks.generate_adv_samples import generate_one_adv_sample
from llava.model import *
from torch.utils.data import DataLoader
from accelerate import Accelerator

accelerator = Accelerator()

def generate_adversarials_pgd(model, args):

    dataset = BaseDataset(dataset=args.dataset, model=model, path=args.adv_path, task=args.task, image_size=args.image_size, subset=args.subset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=False, pin_memory=True)

    model, dataloader = accelerator.prepare(model, dataloader)

    acc1, acc5 = AverageMeter(), AverageMeter()
    for data in tqdm(dataloader):
        images, base_paths, labels = data
        # images = images.cuda().half()
        # labels = labels.cuda()
        images = images.half()
        labels = labels.half()

        if args.targeted:
            targeted_labels = torch.cat([get_different_class(c_true, dataset.label_list) for c_true in labels])
        else:
            targeted_labels = None
        if args.attack_name and args.attack_name != 'None':
            images = generate_one_adv_sample(
                    args,
                    images, 
                    model,
                    dataset.descriptors_embeds if args.attack_descriptors else dataset.text_label_embeds, 
                    save_names=base_paths, 
                    y=targeted_labels,
                    n_classes=len(dataset.label_list))
        
        with torch.no_grad():

            logits = model(images, dataset.descriptors_embeds if args.use_descriptors else dataset.text_label_embeds)
            _acc1, _acc5 = accuracy(logits, labels, topk=(1, 5))

            acc1.update(_acc1[0].item())
            acc5.update(_acc5[0].item())

    print("** Acc@1: {} Acc@5: {} **".format(acc1.avg, acc5.avg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="['openai/clip-vit-large-patch14']")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "caption"])
    parser.add_argument("--dataset", type=str, default="['coco']")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--save_image", type=boolean_string, required=False, default='True')
    parser.add_argument("--adv_path", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=None)

    # whether to use descriptors for Imagenet label retrieval
    parser.add_argument("--use_descriptors", type=boolean_string, default='False')
    # whether to attack descriptors or attack plain labels and use descriptors to do prediction
    parser.add_argument("--attack_descriptors", type=boolean_string, default='False')

    # args for adv attack
    parser.add_argument("--attack_name", type=str, default="pgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--eps", type=str, default="[0.2]")
    parser.add_argument("--norm", type=float, default=np.inf)
    parser.add_argument("--use_last_n_hidden", type=int, default=1)
    parser.add_argument("--targeted", type=boolean_string, default='False')

    # sparse l1 descent
    parser.add_argument("--grad_sparsity", type=int, default=99, help='grad sparsity for sparse_l1_descent')

    # Carlini Wagner attack
    parser.add_argument("--binary_search_steps", type=int, default=5)
    parser.add_argument("--initial_const", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0)

    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', None)
    print('Slurm Job ID: ', SLURM_JOB_ID)
    args = parser.parse_args()

    if args.attack_descriptors and not args.use_descriptors:
        raise ValueError("If attack_descriptors is True, use_descriptors must be True")

    eps = eval(args.eps)
    model_names = eval(args.model_name)
    datasets = eval(args.dataset)
    for model_name in model_names:
    # replace with automodel?
        torch.cuda.empty_cache()
        args.model_name = model_name
        model = get_model(args)

        for ds in datasets:
            args.dataset = ds
            if ds != 'imagenet' and args.use_descriptors:
                raise ValueError("Only imagenet dataset has descriptors")
            for ep in eps:
                args.eps = ep

                print("==> Generating adversarial datasets with args: dataset: {}, model_name: {}, attack: {}, eps: {}".format(ds, model_name, args.attack_name, ep))
                torch.cuda.empty_cache()
                args.save_folder = f"/groups/sernam/adv_llava/adv_datasets/{ds}/{SLURM_JOB_ID}_{'clip'+str(args.image_size)}_{args.attack_name}_eps{ep}_nbiter{args.nb_iter}{'_targeted' if args.targeted else ''}{'_attack_descriptors' if args.attack_descriptors else ''}"
                assert not os.path.exists(args.save_folder), f"Folder {args.save_folder} already exists. It's likely the targeted adversarial dataset is already created."
                os.makedirs(args.save_folder)
                # post-process args
                if args.attack_name == 'None':
                    args.lr = None
                    args.nb_iter = None
                    args.norm = None
                    args.eps = None
                    args.grad_sparsity = None
                    args.binary_search_steps = None
                    args.confidence = None
                    args.initial_const = None
                    args.targeted = None
                
                elif args.attack_name == 'pgd':
                    args.grad_sparsity = None
                    args.binary_search_steps = None
                    args.confidence = None
                    args.initial_const = None
                elif args.attack_name == 'sl1d':
                    args.grad_sparsity = None
                    args.norm = None
                    args.binary_search_steps = None
                    args.confidence = None
                    args.initial_const = None
                elif args.attack_name == 'cw':
                    args.grad_sparsity = None
                    args.eps = None
                    args.norm = None
                else:
                    raise NotImplementedError(f"Attack {args.attack_name} not implemented")

                for k,v in vars(args).items():
                    print(f"{k}: {v}")

                generate_adversarials_pgd(model, args)

                print("=" * 50)



    