import torch
import torch.utils.data as dutils
from typing import Any, List, Union
import transformers
from torch.utils.data import Dataset
from PIL import Image
import os, json
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModelWithProjection, AutoProcessor
from llava.mm_utils import process_images

from utils.func import make_descriptor_sentence

COCO_CLS = '/groups/sernam/datasets/coco/coco_test.txt'
COCO_2014CLS = '/groups/sernam/datasets/coco/coco_2014val.txt'
COCO_CAP = '/groups/sernam/datasets/coco/coco_test.json'
IMAGENET_CLS = '/groups/sernam/datasets/imagenet/labels/imagenet_test.txt'

IMAGENET_DES = '/groups/sernam/datasets/imagenet/labels/descriptors_imagenet.json'
path_config = {'coco': {'classification': COCO_2014CLS, 'caption': COCO_CAP}, 'imagenet': {'classification': IMAGENET_CLS, 'descriptors': IMAGENET_DES}}
LLAVA_VQAV2 = '/groups/sernam/datasets/vqa/vqav2/coco2014val_questions_llava.jsonl'




def read_file(path: str):
    if path.endswith('.txt'):
        with open(path) as f:
            return f.readlines()
    elif path.endswith('.json') or path.endswith('.jsonl'):
        with open(path) as f:
            return json.load(f)
    else:
        raise NotImplementedError("Only support .txt and .json files")

class BaseDataset(Dataset):
    def __init__(self, 
                 dataset = 'coco', 
                 model: Any = None,
                 model_name: str = None,
                 image_processor: Any = None,
                 tokenizer: Any = None,
                 path: str = None, 
                 task: str= 'classification', 
                 subset = None, 
                 use_descriptors = False):
        """
        path: path to folder containing all targeted image. This is used to replace the folder path with 
            the target folder (e.g. the adversarial folder)
        """


        self.data_list = np.array(read_file(path_config[dataset][task])) 
        if path is not None and path != 'None':
            self.data_list = [os.path.join(path, os.path.basename(line).split('|')[0]) for line in self.data_list]
        self.use_descriptors = use_descriptors
        if task == 'classification':
            if dataset == 'coco':
                self.label_list = set()
                for line in self.data_list:
                    label = line.split('|')[-1].split('\n')[0]
                    self.label_list.add(label)
                self.label_list = sorted(list(self.label_list))
            elif dataset == 'imagenet':
                with open('/groups/sernam/datasets/imagenet/labels/imagenet_simple_labels.json') as f:
                    self.label_list = json.load(f)
        if subset:
            self.data_list = np.random.permutation(self.data_list)[:subset]
        if use_descriptors:
            with open(path_config[dataset]['descriptors']) as f:
                self.descriptors = json.load(f)
        self.task = task
        self.model = model
        self.text_label_embeds = self._encode_labels()
        self.descriptors_embeds = self._encode_descriptors() if use_descriptors else None

        # image_processor and tokenizer are used to be compatible with LLaVA v1.5
        self.image_processor = image_processor 
        self.tokenizer = tokenizer
        
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        

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
        if 'llava-v1.5' in self.model_name:
            image = process_images([image], self.image_processor, self.model.model_config)[0]
        else:
            image = self.processor(images=image, return_tensors='pt')['pixel_values'][0]
                
        label = self._load_label(idx) if self.task == 'classification' else self._load_caption(idx)
        # print(label)
        return image, base_path, label
    
    @torch.no_grad()
    def _encode_labels(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        return a tensor of shape (N_labels, 768)
        """
        print("==> Loading text label embeddings...")
        text_labels = ["a photo of %s"%v for v in self.label_list]
        text_labels = self.tokenizer(text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = F.normalize(self.model.text_model(text_labels).text_embeds, p=2., dim=-1) # (N_labels, 768)

        return text_label_embeds
    

    @torch.no_grad()
    def _encode_descriptors(self):

        print("==> Loading descriptors embeddings...")
        text_label_embeds = []
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained(self.model_name, torch_dtype=torch.float16).cuda().eval()
        for label in self.label_list:
            examples = self.descriptors[label]
            sentences = []
            for example in examples:
                sentence = f"{label} {make_descriptor_sentence(example)}"
                sentences.append(sentence)
            
            text_descriptor = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_descriptor_embeds = F.normalize(text_model(text_descriptor).text_embeds, p=2., dim=-1).mean(0) # (N_descriptions, 768)
            text_label_embeds.append(text_descriptor_embeds) # list of (N_descriptions, 768) len = # N_labels
        text_label_embeds = torch.stack(text_label_embeds)
        del tokenizer
        del text_model
        return text_label_embeds


if __name__ == '__main__':
    dataset = BaseDataset(dataset='imagenet', model_name='asdf', use_descriptors=True)
