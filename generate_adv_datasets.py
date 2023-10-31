
from tqdm import tqdm
import transformers
import json
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
from datasets_loader import CLIPDataset
from models import get_model
from utils.metric import AverageMeter, accuracy
from utils.func import make_descriptor_sentence
from llava.utils import disable_torch_init
from attacks.helpers import * 
from attacks.generate_adv_samples import generate_one_adv_sample
from llava.model import *
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets_loader import get_dataloader
from utils.helpers import get_model_name_from_path
from attacks.torchattacks_wrapper import * 

accelerator = Accelerator()

def generate_adversarials_dataset(model, args):

    model = accelerator.prepare(model)
    dataset, dataloader = get_dataloader(args, model)
    dataloader = accelerator.prepare(dataloader)
    text_label_embeds = dataset.text_label_embeds.cuda()
    model.eval()

    if args.attack_name != "None" and args.attack_name != 'pgd':
        attack = eval(args.attack_name)(model, eps=args.eps, nb_iter=args.nb_iter, norm=args.norm)

    acc1, acc5 = AverageMeter(), AverageMeter()
    adv_acc1, adv_acc5 = AverageMeter(), AverageMeter()
    for data in tqdm(dataloader):
        if  args.task == 'classification' or 'clip' in args.model_path or 'blip2_image_text_matching' in args.model_path:
            images, base_paths, labels = data
            input_ids = dataset.query_input_ids.cuda().long() if dataset.query_input_ids is not None else None
        else:
            input_ids, images = data
            input_ids = input_ids.cuda().long() 

        images = images.half()

        if args.targeted:
            if args.task == 'retrieval':
                y = torch.cat([get_different_class(c_true, [i for i in range(dataset.text_label_embeds.shape[0])]) for c_true in labels])
            else:
                y = torch.cat([get_different_class(c_true, dataset.label_list) for c_true in labels])
        else:
            y = None

        if args.attack_name and args.attack_name != 'None':
            if args.attack_name != 'pgd':
                adv_images = attack(images, y=targeted_labels)
            else:
                adv_images = generate_one_adv_sample(
                        args,
                        images, 
                        model,
                        dataset.text_label_embeds, 
                        save_names=base_paths, 
                        y=y,
                        n_classes=len(dataset.label_list))
        
        with torch.no_grad():

            logits = model(images, text_label_embeds=text_label_embeds, input_ids=input_ids, text=None, is_attack=False)
            
            _acc1, _acc5 = accuracy(logits, labels, topk=(1, 5))

            acc1.update(_acc1[0].item())
            acc5.update(_acc5[0].item())

            if args.attack_name and args.attack_name != 'None':
                logits = model(adv_images, text_label_embeds=text_label_embeds, text=None, is_attack=False)
            
                _advacc1, _advacc5 = accuracy(logits, labels, topk=(1, 5))

                adv_acc1.update(_advacc1[0].item())
                adv_acc5.update(_advacc5[0].item())

    print("** Acc@1: {} Acc@5: {} **".format(acc1.avg, acc5.avg))
    if args.attack_name and args.attack_name != 'None':
        print("** Adv Acc@1: {} Adv Acc@5: {} **".format(adv_acc1.avg, adv_acc5.avg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="['openai/clip-vit-large-patch14']")
    parser.add_argument("--image-folder", type=str, required=False, default=None)
    parser.add_argument("--image_ext", type=str, required=False, default='pt')
    parser.add_argument("--save-folder", type=str, required=False, default=None)
    parser.add_argument("--data-file", type=str, required=False, default=None)
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "retrieval"])
    parser.add_argument("--dataset", type=str, default="['coco']")
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--save_image", type=boolean_string, required=False, default='True')
    parser.add_argument("--batch_size", type=int, required=False, default=1, nargs='?')
    parser.add_argument("--num_workers", type=int, required=False, default=1, nargs='?')
    parser.add_argument("--use_ce_loss", type=boolean_string, default='True')
    parser.add_argument("--prompt_formatter", type=str, default="a photo of {}")
    parser.add_argument("--model-type", type=str, default=None)

    parser.add_argument("--query", type=str, default="")

    # params for decoder
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1_1")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)


    # whether to use descriptors for Imagenet label retrieval
    parser.add_argument("--use_descriptors", type=boolean_string, default='False')
    # whether to attack descriptors or attack plain labels and use descriptors to do prediction
    parser.add_argument("--attack_descriptors", type=boolean_string, default='False')

    ## =============== args for adv attack =================
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
    model_paths = eval(args.model_path)
    datasets = eval(args.dataset)
    for model_path in model_paths:
        torch.cuda.empty_cache()
        args.model_path = model_path
        model = get_model(args).to(accelerator.device)
        model_name = get_model_name_from_path(model_path)

        for ds in datasets:
            args.dataset = ds
            if ds != 'imagenet' and args.use_descriptors:
                raise ValueError("Only imagenet dataset has descriptors")
            for ep in eps:
                args.eps = ep

                print("==> Generating adversarial datasets with args: dataset: {}, model_name: {}, attack: {}, eps: {}"
                        .format(ds, model_name, args.attack_name, ep))
                torch.cuda.empty_cache()

                if args.save_image:
                    if not args.save_folder:
                        args.save_folder = "/groups/sernam/adv_llava/adv_datasets/{}/{}/{}_{}_eps{}_nbiter{}{}{}".format(ds, args.task, 
                                SLURM_JOB_ID, model_name, ep, args.nb_iter, 
                                '_attack_descriptors' if args.attack_descriptors else '', '_targeted' if args.targeted else '')
                        
                    assert not os.path.exists(args.save_folder), f"Folder {args.save_folder} already exists. It's likely the targeted adversarial dataset is already created."
                    os.makedirs(args.save_folder, exist_ok=True)

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

                generate_adversarials_dataset(model, args)

                print("=" * 50)



    