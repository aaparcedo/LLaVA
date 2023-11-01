import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
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
from torch.utils.data import DataLoader

import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

COCO_CLS = '/groups/sernam/datasets/coco/coco_trainval2014.jsonl'
COCO_VQA_IMAGE = '/groups/sernam/datasets/coco/coco_2014val.txt'
COCO_CAP = '/groups/sernam/datasets/coco/coco_test.json'
IMAGENET_CLS = '/groups/sernam/datasets/imagenet_val2012.jsonl'

CIFAR100_CLS = '/groups/sernam/datasets/cifar100/cifar100_test.jsonl'

IMAGENET_DES = '/groups/sernam/datasets/imagenet/labels/descriptors_imagenet.json'
path_config = {'coco': {'classification': COCO_CLS, 'caption': COCO_CAP}, 'imagenet': {'classification': IMAGENET_CLS, 'descriptors': IMAGENET_DES}, 'cifar100': {'classification': CIFAR100_CLS}}
LLAVA_VQAV2 = '/groups/sernam/datasets/vqa/vqav2/coco2014val_questions_llava.jsonl'


def read_file(path: str):
    if path.endswith('.txt'):
        with open(path) as f:
            return f.readlines()
    elif path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    elif path.endswith('.jsonl'):
        with open(path) as f:
            return [json.loads(q) for q in f]
    else:
        raise NotImplementedError("Only support .txt, .jsonl and .json files")


class BaseDataset(Dataset):
    def __init__(self,
                dataset = 'coco',
                task = 'classification',
                data_file = None,
                image_folder = None,
                subset = None,
                use_descriptors = False,
                ):
        self.data_file = data_file if data_file else path_config[dataset][task]
        self.data_list = np.array(read_file(self.data_file))
        self.task = task
        self.image_folder = image_folder

        # get a sorted unique label list. 
        self.label_list = sorted(list(set([line['text'] for line in self.data_list])))

        if subset:
            self.data_list = np.random.permutation(self.data_list)[:subset]

        if use_descriptors:
            with open(path_config[dataset]['descriptors']) as f:
                self.descriptors = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        
        file_path = os.path.join(self.image_folder, self.data_list[id]["image"]) if self.image_folder else self.data_list[id]["image"]
        return Image.open(file_path).convert("RGB"), self.data_list[id]["image"]

    
    def __getitem__(self, index) -> Any:
        raise NotImplementedError()

class CLIPDataset(BaseDataset):
    def __init__(self, 
                dataset = 'coco',
                task = 'classification',
                image_folder = None,
                data_file = None,
                subset = None,
                use_descriptors = False,
                model: Any = None):
        
        """
        ** Dataset for both CLIP and original LLaVA **
        :param dataset: 'coco' or 'imagenet'
        :param model: model object
        """
        super().__init__(dataset=dataset, task=task, image_folder=image_folder, subset=subset, data_file=data_file)

        self.task = task
        self.model = model
        self.text_label_embeds = self._encode_labels() if task != 'vqa' else None 
        self.descriptors_embeds = self._encode_descriptors() if use_descriptors else None

        
    def _load_label_or_text(self, id: int) -> Union[torch.Tensor, str]:
        # in case of classification, return the label index
        # in case of retrieval, return the string text corresponding to this image
        if self.task == 'retrieval':    
            return self.data_list[id]['text']
        elif self.task == 'classification':
            label_name = self.data_list[id]['text']
            label = self.label_list.index(label_name)
            return label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image = self.model.processor(images=image, return_tensors='pt')['pixel_values'][0]
        label = self._load_label_or_text(idx) 
        # print(label)
        return image, base_path, label
    
    @torch.no_grad()
    def _encode_labels(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        return a tensor of shape (N_labels, 768). This is for encoding all the classification class labels,
        so we don't have to recompute for every image.
        """
        print("==> Loading text label embeddings...")
        text_labels = ["a photo of %s"%v for v in self.label_list]
        text_labels = self.model.processor(text=text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = F.normalize(self.model.text_encoder(text_labels).text_embeds, p=2., dim=-1) # (N_labels, 768)

        return text_label_embeds

    # @torch.no_grad()
    # def _encode_descriptors(self):

    #     print("==> Loading descriptors embeddings...")
    #     text_label_embeds = []
    #     for label in self.label_list:
    #         examples = self.descriptors[label]
    #         sentences = []
    #         for example in examples:
    #             sentence = f"{label} {make_descriptor_sentence(example)}"
    #             sentences.append(sentence)
            
    #         text_descriptor = self.model.processor(text=sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
    #         text_descriptor_embeds = F.normalize(self.model.text_model(text_descriptor).text_embeds, p=2., dim=-1).mean(0) # (N_descriptions, 768)
    #         text_label_embeds.append(text_descriptor_embeds) # list of (N_descriptions, 768) len = # N_labels
    #     text_label_embeds = torch.stack(text_label_embeds)
    #     return text_label_embeds
    

    @torch.no_grad()
    def _encode_captions(self):
        pass

def get_imagenet_data():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    class_idx = json.load(open("./data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    imagnet_data = image_folder_custom_label(root='./data/imagenet', 
                                             transform=transform,
                                             idx2label=idx2label)
    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
    print("Used normalization: mean=", MEAN, "std=", STD)
    return iter(data_loader).next()

def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()

def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    
def image_folder_custom_label(root, transform, idx2label) :
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2


@torch.no_grad()
def get_accuracy(model, data_loader, atk=None, n_limit=1e10, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in data_loader:

        X = images.to(device)
        Y = labels.to(device)

        if atk:
            X = atk(X, Y)

        pre = model(X)

        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break

    return (100 * float(correct) / total)
