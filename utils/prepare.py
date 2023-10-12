from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, \
                         BlipModel, BlipForImageTextRetrieval, AutoProcessor, BlipForQuestionAnswering
# from ..llava.model import *
import torch
from PIL import Image
import requests


def get_model(model_name, task):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == 'clip':
        vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
        text_model = CLIPTextModelWithProjection.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
    elif model_name == 'blip':
        model = BlipForImageTextRetrieval.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
        vision_model = model.vision_model
        text_model = model.text_model

    return tokenizer, vision_model, text_model


if __name__ == '__main__':
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)

    text = "an image of a cat"
    text_image_inputs = processor(images=image, text=text, return_tensors="pt")

    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    outputs = model(**text_image_inputs)