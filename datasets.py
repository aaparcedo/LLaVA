import torch
import torch.utils.data as dutils
from typing import List, Union
import transformers
from torch.utils.data import Dataset
from PIL import Image
import os, json
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from utils.func import make_descriptor_sentence

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda().eval()

COCO_CLS = '/groups/sernam/datasets/coco/coco_test.txt'
COCO_CAP = '/groups/sernam/datasets/coco/coco_test.json'
IMAGENET_CLS = '/groups/sernam/datasets/imagenet/labels/imagenet_test.txt'

IMAGENET_DES = '/groups/sernam/datasets/imagenet/labels/descriptors_imagenet.json'
path_config = {'coco': {'classification': COCO_CLS, 'caption': COCO_CAP}, 'imagenet': {'classification': IMAGENET_CLS, 'descriptors': IMAGENET_DES}}

def read_file(path: str):
    if path.endswith('.txt'):
        with open(path) as f:
            return f.readlines()
    elif path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    else:
        raise NotImplementedError("Only support .txt and .json files")

class BaseDataset(Dataset):
    def __init__(self, dataset = 'coco', 
                 path: str = None, 
                 task: str= 'classification', 
                 image_size = 224, 
                 subset = None, 
                 use_descriptors = False):
        """
        path: path to folder containing all targeted image. This is used to replace the folder path with 
            the target folder (e.g. the adversarial folder)
        """

        self.data_list = read_file(path_config[dataset][task])
        if path:
            self.data_list = [os.path.join(path, os.path.basename(line).split('|')[0]) for line in self.data_list]
        self.use_descriptors = use_descriptors
        if task == 'classification':
            self.label_list = set()
            for line in self.data_list:
                label = line.split('|')[-1].split('\n')[0]
                self.label_list.add(label)
            self.label_list = sorted(list(self.label_list))
        if subset:
            self.data_list = self.data_list[:subset]
        if use_descriptors:
            with open(path_config[dataset]['descriptors']) as f:
                self.descriptors = json.load(f)
        self.task = task
        self.transform = transforms.Compose([
                        transforms.Resize(size=(image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                    ])
        self.text_label_embeds = self._encode_labels()
        
        

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        if self.task == 'caption':
            return Image.open(self.data_list[id]["filename"]).convert("RGB"), self.data_list[id]["filename"]
        else:
            return Image.open(self.data_list[id].split('|')[0]).convert("RGB"), self.data_list[id].split('|')[0]

    def _load_caption(self, id: int) -> List[str]:
        if self.task == 'caption':
            return self.data_list[id]["captions"][:5]
        else:
            return None
    
    def _load_label(self, id: int) -> Union[torch.Tensor, str]:
        if self.task == 'caption':    
            return torch.tensor([int(bit) for bit in self.data_list[id]["label"]], dtype=torch.float32)
        else:
            label_name = self.data_list[id].split('|')[-1].split('\n')[0]
            label = self.label_list.index(label_name)
            return label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image = self.transform(image)
        label = self._load_label(idx) if self.task == 'classification' else self._load_caption(idx)
        # print(label)
        return image, base_path, label
    
    @torch.no_grad()
    def _encode_labels(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        When use_descriptors is False, return a tensor of shape (N_labels, 768)
        When use_descriptors is True, return a list of tensors of shape (N_descriptions, 768)
        """
        print("==> Loading text label embeddings...")
        if self.use_descriptors:
            text_label_embeds = []

            for label in self.label_list:
                examples = self.descriptors[label]
                sentences = []
                for example in examples:
                    sentence = f"{label} {make_descriptor_sentence(example)}"
                    sentences.append(sentence)
                
                text_descriptor = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_descriptor_embeds = F.normalize(text_model(text_descriptor).text_embeds, p=2., dim=-1) # (N_descriptions, 768)
                text_label_embeds.append(text_descriptor_embeds) # list of (N_descriptions, 768) len = # N_labels
        else:
            text_labels = ["a photo of %s"%v for v in self.label_list]
            text_labels = tokenizer(text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
            text_label_embeds = F.normalize(text_model(text_labels).text_embeds, p=2., dim=-1) # (N_labels, 768)

        return text_label_embeds

if __name__ == '__main__':
    dataset = BaseDataset(dataset='imagenet', task='classification', use_descriptors=True)
