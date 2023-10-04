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


COCO_CLS = '/groups/sernam/datasets/coco/coco_test.txt'
COCO_CAP = '/groups/sernam/datasets/coco/coco_test.json'


IMAGENET_CLS = '/groups/sernam/datasets/imagenet/imagenet_test.txt'

class BaseDataset(Dataset):
    def __init__(self, dataset, task='classification', image_size = 224):
        if dataset == 'coco':
            if task == 'classification':
                with open(COCO_CLS) as f:
                    self.data_list = json.load(f)
            elif task == 'caption':
                with open(COCO_CAP) as f:
                    self.data_list = json.load(f)
        elif dataset == 'imagenet':
            if task == 'classification':
                with open(IMAGENET_CLS) as f:
                    self.data_list = f.readlines()
        else:
            raise NotImplementedError("Path format not implemented. Need to be either .txt-->classification or .json-->captioning")
        
        self.task = task
        self.transform = transforms.Compose([
                        transforms.Resize(size=(image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                    ])

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        if self.task == 'caption':
            return Image.open(self.data_list[id]["filename"]).convert("RGB")
        else:
            return Image.open(self.data_list[id].split('|')[0]).convert("RGB")

    def _load_caption(self, id: int) -> List[str]:
        if self.task == 'caption':
            return self.data_list[id]["captions"][:5]
        else:
            return None
    
    def _load_label(self, id: int) -> Union[torch.Tensor, str]:
        if self.task == 'caption':    
            return torch.tensor([int(bit) for bit in self.data_list[id]["label"]], dtype=torch.float32)
        else:
            return self.data_list[id].split('|')[-1].split('\n')[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.transform(self._load_image(idx))
        label = self._load_label(idx)
        caption = self._load_caption(idx)

        return image, label, caption