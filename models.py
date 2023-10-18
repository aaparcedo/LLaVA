import os
from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, \
                         BlipModel, Blip2ForConditionalGeneration, CLIPModel, BlipForImageTextRetrieval, AutoProcessor, BlipForQuestionAnswering
                        
from llava.model import *
import torch
from PIL import Image
import requests
import torch.nn.functional as F
import torch.nn as nn
from run_llava import run_LLaVA
from llava.modelv2.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda()
        self.text_model = CLIPTextModelWithProjection.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda()
        self.processor = AutoProcessor.from_pretrained(args.model_name)

    def forward(self, images, text_label_embeds, is_attack=False):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """

        image_embeds = F.normalize(self.vision_model(images).image_embeds, p=2., dim=-1)
                ## zero-shot result with clip
        logits = torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label
        return logits
    
class LLaVA(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.llava_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.llava_model = LlavaLlamaForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda().eval()
        self.processor = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda()
        self.text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()

    def forward(self, images, text_label_embeds, is_attack=False):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """

        if is_attack:
            # for attack, we only attack the clip part of llava
            image_embeds = F.normalize(self.clip_vision_model(images).image_embeds, p=2., dim=-1)
                    ## zero-shot result with clip
            logits = torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label
            return logits

        responses, text_response_embeds = [], []
        for image in images:
            response, _ = run_LLaVA(self.args, self.llava_model, self.llava_tokenizer, image.unsqueeze(0))
            responses.append(response)
            text_response = self.clip_tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_response_embed = F.normalize(self.clip_text_model(text_response).text_embeds, p=2., dim=-1)
            text_response_embeds.append(text_response_embed)

        text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768

        # else:
        logits = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=1)
        return logits


class LLaVA2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(args.m, args.model_base, model_name)
    
    def forward(self, images, text_label_embeds, is_attack=False):
        
    


class BLIP2(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda().eval()
        self.clip_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()

        self.processor = AutoProcessor.from_pretrained(args.model_name)
        self.query = self.processor(text=args.query, padding=True, truncation=True, return_tensors="pt")['input_ids'].to('cuda')
        # TODO: only attack blip's vision encoder.
    def forward(self, images, text_label_embeds, is_attack=False):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """
        
        # processed_images = self.processor(images=images, return_tensors="pt")['pixel_values'].cuda().half()
        text_response_embeds = []
        for image in images:
            generated_ids = self.model.generate(pixel_values=image.unsqueeze(0)) # add a conditional query to continue from the query
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            text_response = self.clip_tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_response_embed = F.normalize(self.clip_text_model(text_response).text_embeds, p=2., dim=-1)
            text_response_embeds.append(text_response_embed)

        text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768
        logits = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=1)
        return logits


def get_model(args):
    if 'llava' in args.model_name.lower():
        model = LLaVA(args)
    elif 'clip' in args.model_name.lower():
        model = CLIP(args)
    elif 'blip2' in args.model_name.lower():
        model = BLIP2(args)
    
    return model

if __name__ == '__main__':
    llava = LLaVA()