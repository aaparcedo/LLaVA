import torch
import torch.utils.data as dutils
from typing import List
import transformers
from torch.utils.data import Dataset
from PIL import Image
import os, json
import numpy as np
import torch.nn.functional as F


# Change these to path of local COCO dataset:
query_list = '/groups/sernam/datasets/coco/coco_test.json'

with open(query_list) as f:
    query_list = json.load(f)

clip_model_name = "openai/clip-vit-large-patch14"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained(clip_model_name, output_hidden_states=True).to(device).eval()
language_model = transformers.CLIPTextModelWithProjection.from_pretrained(clip_model_name).to(device).eval()

class CLIPModel(torch.nn.Module):
    def __init__(self, vision_model, language_model):
        super().__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07))

    def forward_image(self, image):
        x_i = self.vision_model(image).image_embeds        
        return F.normalize(x_i)
    
    def forward_text(self, text):
        x_t = self.language_model(text).text_embeds
        return F.normalize(x_t)

CLIP = CLIPModel(vision_model, language_model)
# CLIP.load_state_dict(torch.load('out/coco_old_gallery.pth'), strict=True)

processor = transformers.AutoProcessor.from_pretrained(clip_model_name)


class CoCo(Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, id: int):
        return Image.open(self.data_list[id]["filename"]).convert("RGB")

    def _load_caption(self, id: int) -> List[str]:
        return self.data_list[id]["captions"][:5]
    
    def _load_label(self, id: int) -> torch.Tensor:
        return torch.tensor([int(bit) for bit in self.data_list[id]["label"]], dtype=torch.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self._load_image(idx)
        caption = self._load_caption(idx)
        output = self.processor(text=caption, images=image, return_tensors="pt", truncation = True, padding = "max_length")
        image = output['pixel_values'][0]
        caption = output['input_ids']

        label = self._load_label(idx)

        return image, caption, label

# Encodes all text and images in a dataset
def encode_both(clip_model, dataset: dutils.Dataset, batch_size = 16):
    with torch.no_grad():
        image_to_text_map = []
        text_to_image_map = []

        dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0

        for images, text, _ in dataloader:
            images = images.to(device)
            text = text.to(device)

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = text.shape

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            
            image_encodings.append(clip_model.forward_image(images))
            text_encodings.append(clip_model.forward_text(text))

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map
    

# Encodes all text and images in a dataset
def encode_image(clip_model, dataset: dutils.Dataset, batch_size = 16):
    with torch.no_grad():

        dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        image_encodings = []
        label_all = []

        for images, _, label in dataloader:
            images = images.to(device)
            label = label.to(device)

            image_encodings.append(clip_model.forward_image(images))
            label_all.append(label)

        image_encodings = torch.cat(image_encodings)
        label_all = torch.cat(label_all)

        return image_encodings, label_all


def recall_at_k(image_encodings, text_encodings, text_to_image_map, image_to_text_map, k_vals: List[int]):
     
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    dist_matrix = text_encodings @ image_encodings.T 
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


query_dataset = CoCo(query_list, processor)

k_vals=[1, 5, 10, 50]

image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_both(CLIP, query_dataset)

t2i, i2t = recall_at_k(image_encodings, text_encodings, text_to_image_map, image_to_text_map, k_vals=k_vals)

print("Text-to-image Recall@K")
for k, x in zip(k_vals, t2i):
    print(f" R@{k}: {100*x:.2f}%")

print("Image-to-text Recall@K")
for k, x in zip(k_vals, i2t):
    print(f" R@{k}: {100*x:.2f}%")