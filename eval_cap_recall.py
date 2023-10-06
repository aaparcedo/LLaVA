import argparse
import torch
import torch.utils.data as dutils
from typing import List
from tqdm import tqdm
import transformers
from torch.utils.data import Dataset
from PIL import Image
import os, json
import numpy as np
import torch.nn.functional as F
from attacks.generate_adv_samples import generate_one_adv_sample
from attacks.helpers import boolean_string
from run_llava import run_LLaVA
from llava.utils import disable_torch_init
from llava.model import *
from datetime import datetime

from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoProcessor

clip_model_name = "openai/clip-vit-large-patch14"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name, output_hidden_states=True, torch_dtype=torch.float16).to(device).eval()
language_model = CLIPTextModelWithProjection.from_pretrained(clip_model_name, torch_dtype=torch.float16).to(device).eval()

model_name = "/groups/sernam/ckpts/LLAMA-on-LLaVA"
# llava_tokenizer = AutoTokenizer.from_pretrained(model_name)
# llava_model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).to(device).eval()

processor = AutoProcessor.from_pretrained(clip_model_name)

class CoCo(Dataset):
    def __init__(self, data_list, processor, base_path=None):
        """
        base_path: path to folder containing all targeted image. This is used to replace the folder path with
            the target folder (e.g. the adversarial folder)
        """
        self.data_list = data_list
        self.processor = processor
        if base_path:
            for dict_ in self.data_list:
                dict_["filename"] = os.path.join(base_path, os.path.basename(dict_["filename"]))

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        # return rgb image and its base filename
        return Image.open(self.data_list[id]["filename"]).convert("RGB"), os.path.basename(self.data_list[id]["filename"])

    def _load_caption(self, id: int) -> List[str]:
        return self.data_list[id]["captions"][:5]
    
    def _load_label(self, id: int) -> torch.Tensor:
        return torch.tensor([int(bit) for bit in self.data_list[id]["label"]], dtype=torch.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, basename = self._load_image(idx)
        caption = self._load_caption(idx)
        output = self.processor(text=caption, images=image, return_tensors="pt", truncation = True, padding = "max_length")
        image = output['pixel_values'][0]
        caption = output['input_ids']

        label = self._load_label(idx)

        return image, caption, label, basename


def encode_all_captions(dataset: dutils.Dataset, batch_size = 1):
    dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    text_label_encodings = []

    for _, text, _, _ in tqdm(dataloader):
        text = text.to(device)

        batch_size= text.shape[0]

        with torch.no_grad():
            text = torch.flatten(text, start_dim=0, end_dim=1)
            text_label_encoding = language_model(text).text_embeds 

        text_label_encodings.append(text_label_encoding)
    text_label_embeds = torch.cat(text_label_encodings) 
    return text_label_embeds


# Encodes all text and images in a dataset
def encode_llava(args, text_label_embeds, dataset: dutils.Dataset, batch_size = 1):

    """
    batch_size: please set to 1 for now as LlaVa is not batched.
    """

    image_to_text_map = []
    text_to_image_map = []

    dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    image_encodings = []
    text_response_encodings = []

    text_index = 0
    image_index = 0

    for images, text, _, basename in tqdm(dataloader):
        images = images.to(device)
        text = text.to(device)

        # text has shape B x 5 x 77
        batch_size, captions_per_image, _ = text.shape

        if args.attack_name != 'None':
            images = generate_one_adv_sample(images, 
                args.attack_name, 
                text_label_embeds, 
                vision_model, 
                use_descriptors=args.use_descriptors,
                save_image=args.save_image, 
                save_folder=args.adv_path,
                save_names=basename, 
                eps=args.eps, 
                lr=args.lr, 
                nb_iter=args.nb_iter, 
                norm=args.norm)

        # Update text_to_image_map and image_to_text_map for this batch
        for i in range(batch_size):
            # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
            text_indices = list(range(text_index, text_index + captions_per_image))
            image_to_text_map.append(text_indices)
            text_index += captions_per_image

            # Each of the next captions_per_image text captions correspond to the same image
            text_to_image_map += [image_index] * captions_per_image
            image_index += 1

        with torch.no_grad():
            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            #response, img_cls_token = run_LLaVA(args, llava_model, llava_tokenizer, images)

            if args.first_response_only:
                response = response[0]
            #image_embeds = vision_model.visual_projection(img_cls_token) # b 768
            image_embeds = vision_model(images).image_embeds # b 768
            # text_response = clip_tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda() # N_response, embed
            # text_response_encoding = language_model(text_response).text_embeds # 5 768

        image_encodings.append(image_embeds)
        # text_response_encodings.append(text_response_encoding)

    image_encodings = torch.cat(image_encodings)
    # text_response_encodings = torch.stack(text_response_encodings, 0)
    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

    # Normalise encodings
    image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
    # text_response_encodings = text_response_encodings / text_response_encodings.norm(dim=-1, keepdim=True)

    return image_encodings, text_response_encodings, text_to_image_map, image_to_text_map


def recall_at_k(input_encodings, text_encodings, text_to_image_map, image_to_text_map, k_vals: List[int]):
     
    num_text = text_encodings.shape[0]
    num_im = input_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    dist_matrix = text_encodings @ input_encodings.T 

    dist_matrix = dist_matrix.cpu()

    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    for k in k_vals:
        topk = inds[:, :k]
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    dist_matrix = dist_matrix.T
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    for k in k_vals:
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#

    return text_to_image_recall, image_to_text_recall


def main(args):
    start_time = datetime.now()
    query_list = '/groups/sernam/datasets/coco/coco_test.json'

    if args.attack_name != 'None':
        adv_img_folder = f"/groups/sernam/adv_llava/adv_datasets/{args.dataset}/{args.task}_{args.attack_name}_lr{args.lr}_steps{args.nb_iter}"
        if args.attack_name == 'pgd':
            adv_img_folder += f"_eps{args.eps}_norm{args.norm}"
        elif args.attack_name == 'sparse_l1_descent':
            adv_img_folder += f"_eps{args.eps}_grad_sparsity{args.grad_sparsity}"
        else:
            adv_img_folder += f"_bss{args.binary_search_steps}"
        # check if the corresponding adv image folder already exist
        # adv_path != None means the target adverarial dataset is already generated
        if os.path.exists(adv_img_folder):
            args.adv_path = adv_img_folder
        else:
            if args.save_image:
                os.makedirs(adv_img_folder)
            args.adv_path=None
    else:
        args.adv_path = None

    for k,v in vars(args).items():
        print(f"{k}: {v}")

    with open(query_list) as f:
        query_list = json.load(f)
    if args.subset:
        query_list = query_list[:args.subset]
    query_dataset = CoCo(query_list, processor, base_path=args.adv_path)

    k_vals=[1, 5, 10, 50]

    text_label_embeds = encode_all_captions(query_dataset)
    image_encodings, text_response_encodings, text_to_image_map, image_to_text_map = encode_llava(args, text_label_embeds, query_dataset, batch_size=1)

    # if not args.first_response_only:
    #     text_response_encodings = text_response_encodings.mean(dim=1) # llava has multiple responses per image, average them for now
    t2i, i2t = recall_at_k(image_encodings, text_label_embeds, text_to_image_map, image_to_text_map, k_vals=k_vals)
    # cap2llava, llava2cap = recall_at_k(text_response_encodings, text_label_embeds, text_to_image_map, image_to_text_map, k_vals=k_vals)

    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, i2t):
        print(f" R@{k}: {100*x:.2f}%")

    # print("Text-to-Llava Recall@K")
    # for k, x in zip(k_vals, cap2llava):
    #     print(f" R@{k}: {100*x:.2f}%")

    # print("Llava-to-text Recall@K")
    # for k, x in zip(k_vals, llava2cap):
    #     print(f" R@{k}: {100*x:.2f}%")

    end_time = datetime.now()
    time_elapsed = end_time - start_time
    with open(f"/groups/sernam/adv_llava/results/retrieval_results.txt", "a+") as f:
        f.write('='*50+'\n')
        for k,v in vars(args).items():
            f.writelines(f"{k}: {v}\n")
        f.writelines("Text-to-image Recall@K\n")
        for k, x in zip(k_vals, t2i):
            f.writelines(f" R@{k}: {100*x:.2f}%\n")

        f.writelines("Image-to-text Recall@K\n")
        for k, x in zip(k_vals, i2t):
            f.writelines(f" R@{k}: {100*x:.2f}%\n")

        # f.writelines("Text-to-Llava Recall@K")
        # for k, x in zip(k_vals, cap2llava):
        #     f.writelines(f" R@{k}: {100*x:.2f}%\n")

        # f.writelines("Llava-to-text Recall@K\n")
        # for k, x in zip(k_vals, llava2cap):
        #     f.writelines(f" R@{k}: {100*x:.2f}%\n")
        f.write("Start time: {} End time: {} Time elapsed: {}\n".format(start_time, end_time, time_elapsed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/groups/sernam/ckpts/LLAMA-on-LLaVA")
    parser.add_argument("--task", type=str, default="caption")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None, help="number of images to test")
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')
    parser.add_argument("--first_response_only", type=boolean_string, required=False, default='True')

    # args for adv attack
    parser.add_argument("--attack_name", type=str, default="pgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--grad_sparsity", type=int, default=99, help='grad sparsity for sparse_l1_descent')
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--norm", type=float, default=np.inf)
    parser.add_argument("--binary_search_steps", type=int, default=5)
    
    parser.add_argument("--use_descriptors", type=boolean_string, default='False')

    parser.add_argument("--query", type=str, default="Generate five captions for the image.")

    args = parser.parse_args()

    # post-process args
    if args.attack_name == 'None':
        args.lr = None
        args.nb_iter = None
        args.norm = None
        args.eps = None
        args.grad_sparsity = None
        args.binary_search_steps = None
    
    if args.attack_name == 'pgd':
        args.grad_sparsity = None
        args.binary_search_steps = None
    elif args.attack_name == 'sl1d':
        args.grad_sparsity = None
        args.norm = None
        args.binary_search_steps = None
    elif args.attack_name == 'cw2':
        args.grad_sparsity = None
        args.eps = None
        args.norm = None

    main(args)