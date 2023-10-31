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
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, CLIPTextModelWithProjection, AutoProcessor
from llava.mm_utils import process_images, text_to_input_ids
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from utils.func import make_descriptor_sentence
from torch.utils.data import DataLoader

COCO_CLS = '/groups/sernam/datasets/coco/coco_trainval2014.jsonl'
COCO_VQA_IMAGE = '/groups/sernam/datasets/coco/coco_2014val.txt'
COCO_CAP = '/groups/sernam/datasets/coco/coco_test.json'
IMAGENET_CLS = '/groups/sernam/datasets/imagenet_val2012.jsonl'

IMAGENET_DES = '/groups/sernam/datasets/imagenet/labels/descriptors_imagenet.json'
path_config = {'coco': {'classification': COCO_CLS, 'caption': COCO_CAP}, 'imagenet': {'classification': IMAGENET_CLS, 'descriptors': IMAGENET_DES}}
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
                image_ext = 'pt'
                ):
        self.data_file = data_file if data_file else path_config[dataset][task]
        self.data_list = np.array(read_file(self.data_file))
        self.task = task
        self.image_folder = image_folder
        self.image_ext = image_ext



        if task == 'classification':
            if 'coco' in dataset:
                with open('/groups/sernam/datasets/coco/coco_labels_val2014.json', 'r') as f:
                    self.label_list = json.load(f)
        else:
            self.label_list = list(set([line['text'] for line in self.data_list]))

        if subset:
            self.data_list = np.random.permutation(self.data_list)[:subset]
        if use_descriptors:
            with open(path_config[dataset]['descriptors']) as f:
                self.descriptors = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        
        file_path = os.path.join(self.image_folder, self.data_list[id]["image"])
        if self.image_ext == 'pt':
            file_path = os.path.splitext(file_path)[0] + '.pt'
            return torch.load(file_path), self.data_list[id]["image"]

        return Image.open(file_path).convert("RGB"), self.data_list[id]["image"]

    
    def __getitem__(self, index) -> Any:
        raise NotImplementedError()

class CLIPDataset(BaseDataset):
    def __init__(self, 
                dataset = 'coco',
                task = 'classification',
                image_folder = None,
                image_ext = 'pt',
                data_file = None,
                subset = None,
                use_descriptors = False,
                model: Any = None):
        
        """
        ** Dataset for both CLIP and original LLaVA **
        :param dataset: 'coco' or 'imagenet'
        :param model: model object
        """
        super().__init__(dataset=dataset, task=task, image_folder=image_folder, 
                        subset=subset, data_file=data_file, image_ext=image_ext, use_descriptors=use_descriptors)

        self.task = task
        self.model = model
        self.use_descriptors = use_descriptors
        self.text_label_embeds = self.encode_labels()
        self.query_input_ids = None

        
    def _load_label(self, id: int) -> Union[torch.Tensor, str]:
        # in case of classification, return the label index
        # in case of retrieval, return the string text corresponding to this image
        # if self.task == 'retrieval':    
        #     return id
        #     # return self.data_list[id]['text']
        # elif self.task == 'classification':
        label_name = self.data_list[id]['text']
        label = self.label_list.index(label_name)
        return label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        if self.image_ext != 'pt':
            image = self.model.processor(images=image, return_tensors='pt')['pixel_values'][0]
        label = self._load_label(idx) 
        # print(label)
        return image, base_path, label
    
    @torch.no_grad()
    def encode_labels(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        return a tensor of shape (N_labels, 768). This is for encoding all the classification class labels,
        so we don't have to recompute for every image.
        """
        print("==> Loading text label embeddings...")
        if self.task == 'classification':
            if self.use_descriptors:
                text_label_embeds = []
                for label in self.label_list:
                    examples = self.descriptors[label]
                    sentences = []
                    for example in examples:
                        sentence = f"{label} {make_descriptor_sentence(example)}"
                        sentences.append(sentence)
                    text_descriptor = self.model.processor(text=sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                    text_descriptor_embeds = F.normalize(self.model.text_model(text_descriptor).text_embeds, p=2., dim=-1).mean(0) # (N_descriptions, 768)
                    text_label_embeds.append(text_descriptor_embeds) # list of (N_descriptions, 768) len = # N_labels
                text_label_embeds = torch.stack(text_label_embeds)
            else:
                text_labels = ["a photo of %s"%v for v in self.label_list]
                text_labels = self.model.processor(text=text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
                text_label_embeds = self.model.text_encoder(text_labels).text_embeds # (N_labels, 768)
        elif self.task == 'retrieval':
            text_label_embeds = []
            for text in self.label_list:
                text_labels = self.model.processor(text, padding=True, return_tensors="pt")['input_ids'].cuda()
                text_label_embed = self.model.text_encoder(text_labels).text_embeds
                if text_label_embed.shape[0] > 1:
                    text_label_embed = text_label_embed.mean(0)
                text_label_embeds.append(text_label_embed)
            
            text_label_embeds = torch.cat(text_label_embeds)
        print("==> Done.")
        return F.normalize(text_label_embeds, dim=-1)


class LLAVA2Dataset(CLIPDataset):
    def __init__(self, 
                dataset = 'coco',
                task = 'classification',
                image_folder = None,
                data_file = None,
                image_ext = 'pt',
                subset = None,
                use_descriptors = False,
                model: Any = None):
        
        """
        *** Dataset for LLAVA V1.5.
        
        :param dataset: 'coco' or 'imagenet'
        :param model: model object
        """

        super().__init__(dataset=dataset, task=task, 
                         image_folder=image_folder, subset=subset, 
                         data_file=data_file,
                         use_descriptors=use_descriptors, 
                         image_ext=image_ext,
                         model=model)
        self.model = model
        # For classification and retrieval, query is the same across all images
        self.query_input_ids = None if not hasattr(model.args, 'query') else text_to_input_ids(self.model.args.query, self.model).unsqueeze(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        
        image_tensor = image if self.image_ext == 'pt' else self.model.preprocess_image(image)
        #process_images([image], self.model.image_processor, self.model.model.config)[0]
        if self.task == 'classification':
            
            label = self._load_label(idx) 
            # print(label)
            return image_tensor, base_path, label

        else:
            
            if self.task == 'vqa':
                qs = self.data_list[idx]["text"]
                input_ids = text_to_input_ids(qs, self.model)
                return input_ids, image_tensor, self.data_list[idx]["question_id"]

            elif self.task == 'retrieval':
                input_ids = self.query_input_ids
                return input_ids, image_tensor


class BLIP2ITMDataset(BaseDataset):
    def __init__(self, 
                dataset = 'coco',
                task = 'classification',
                image_folder = None,
                data_file = None,
                subset = None,
                use_descriptors = False,
                model: Any = None,
                image_ext = 'pt',
                prompt_formatter: str = None):
        
        """
        *** Dataset for LLAVA V1.5.
        :param dataset: 'coco' or 'imagenet'
        :param model: model object
        """

        super().__init__(dataset=dataset, task=task, 
                         image_folder=image_folder, subset=subset, 
                         data_file=data_file,
                         image_ext=image_ext,
                         use_descriptors=use_descriptors)
        self.model = model
        self.prompt_formatter = prompt_formatter
        self.text_label_embeds = self.encode_labels()
        self.query_input_ids = None


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image_tensor = image if self.image_ext == 'pt' else self.model.vis_processors['eval'](image)
        label_name = self.data_list[idx]['text']
        label = self.label_list.index(label_name)
        # print(label)
        return image_tensor, base_path, label

    def encode_labels(self):
        print("==> Loading text label embeddings...")
        if self.prompt_formatter is not None:
            texts = [self.prompt_formatter.format(l) for l in self.label_list]   
        else:
            texts = [*self.label_list]
        texts = [self.model.text_processors['eval'](t) for t in texts]
        print("Done")
        return self.model.encode_texts(texts)
            
class InstructBLIPDataset(BaseDataset):
    def __init__(self, 
                dataset = 'coco',
                task = 'classification',
                image_folder = None,
                data_file = None,
                subset = None,
                use_descriptors = False,
                image_ext = 'pt',
                model: Any = None):
        
        """
        *** Dataset for LLAVA V1.5.
        :param dataset: 'coco' or 'imagenet'
        :param model: model object
        """

        super().__init__(dataset=dataset, task=task, 
                         image_folder=image_folder, subset=subset, 
                         data_file=data_file,
                         image_ext=image_ext,        
                         use_descriptors=use_descriptors)
        self.model = model
        self.text_label_embeds = self.encode_labels()
        self.query_input_ids = None if not hasattr(model.args, 'query') else model.text_processors(self.model.args.query)['input_ids'].unsqueeze(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, base_path = self._load_image(idx)
        image_tensor = image if self.image_ext == 'pt' else self.model.vis_processors['eval'](image).to(self.model.device).half()

        if self.task == 'classification':
            label = self._load_label(idx) 
            # print(label)
            return image_tensor, base_path, label

        else:
            if self.task == 'vqa':
                qs = self.data_list[idx]["text"]
                if hasattr(self.model.args, 'prompt_format') and self.model.args.prompt_format is not None:
                    qs = self.model.args.prompt_format.format(qs)
                input_ids = self.model.text_processors['eval'](qs)
                return input_ids, image_tensor, self.data_list[idx]["question_id"]

            elif self.task == 'retrieval':
                input_ids = self.query_input_ids[0]
                return input_ids, image_tensor
            
    def encode_labels(self):
        if self.prompt_formatter is not None:
            texts = [self.prompt_formatter.format(l) for l in self.label_list]   
        else:
            texts = [*self.label_list]
        return self.model.encode_texts(texts)


def get_dataloader(args, model):    
    if 'llava-v1.5' in args.model_path:
        dataset = LLAVA2Dataset(dataset=args.dataset, model=model, data_file=args.data_file, image_folder=args.image_folder, task=args.task, subset=args.subset, use_descriptors=args.use_descriptors, image_ext=args.image_ext)
    elif 'llava' in args.model_path or 'clip' in args.model_path:
        dataset = CLIPDataset(dataset=args.dataset, model=model, data_file=args.data_file, image_folder=args.image_folder, task=args.task, subset=args.subset, use_descriptors=args.use_descriptors, image_ext=args.image_ext)
    elif args.model_path == 'blip2_image_text_matching':
        dataset = BLIP2ITMDataset(dataset=args.dataset, model=model, data_file=args.data_file, image_folder=args.image_folder, task=args.task, subset=args.subset, prompt_formatter=args.prompt_formatter, use_descriptors=args.use_descriptors, image_ext=args.image_ext)
    elif 'instructblip' in args.model_path:
        dataset = InstructBLIPDataset(dataset=args.dataset, model=model, data_file=args.data_file, image_folder=args.image_folder, task=args.task, subset=args.subset, use_descriptors=args.use_descriptors, image_ext=args.image_ext)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=False)
    return dataset, dataloader


if __name__ == '__main__':
    dataset = CLIPDataset(dataset='imagenet', use_descriptors=True)
