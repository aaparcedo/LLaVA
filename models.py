from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, \
                         BlipModel, BlipForImageTextRetrieval, AutoProcessor, BlipForQuestionAnswering
from llava.model import *
import torch
from PIL import Image
import requests
import torch.nn.functional as F
import torch.nn as nn
from run_llava import run_LLaVA

class CLIPRetrieval(nn.Module):
    def __init__(self, args, no_grad=True):
        super().__init__(args, no_grad)
        self.args = args
        self.no_grad = no_grad
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda().eval()

    def forward(self, images, text_label_embeds):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """
        if self.no_grad:
            with torch.no_grad():
                image_embeds = F.normalize(self.vision_model(images).image_embeds, p=2., dim=-1)
                logits = torch.matmul(image_embeds, text_label_embeds.t())
        else:
            image_embeds = F.normalize(self.vision_model(images).image_embeds, p=2., dim=-1)
            logits = torch.matmul(image_embeds, text_label_embeds.t())
        return logits
    
class LLaVARetrieval(nn.Module):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.llava_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.llava_model = LlavaLlamaForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda().eval()
        self.clip_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda().eval()

    def forward(self, images, text_label_embeds):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """
        with torch.no_grad():
            text_response_embeds = []
            for image in images:
                response, _ = run_LLaVA(self.args, self.llava_model, self.llava_tokenizer, image.unsqueeze(0))
                text_response = self.clip_tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_response_embed = F.normalize(self.clip_text_model(text_response).text_embeds, p=2., dim=-1)
                text_response_embeds.append(text_response_embed)

            text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768
            logits = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=1)
        return logits


class BLIPLabelRetrieval(nn.Module):
    def __init__(self, args, no_grad=True):
        super().__init__(args)
        self.args = args
        self.blip_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.blip_model = BlipForImageTextRetrieval.from_pretrained(args.model_name, torch_dtype=torch.float16).cuda().eval()
        self.no_grad = no_grad

    def forward(self, images, text_label_embeds):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """

        if self.no_grad:
            with torch.no_grad():
                image_embeds = self.blip_model.vision_model(images).pooler_output
                projected_image_embeds = self.blip_model.vision_proj(image_embeds)
                logits = torch.matmul(projected_image_embeds, text_label_embeds.t())
        else:
            image_embeds = self.blip_model.vision_model(images).pooler_output
            projected_image_embeds = self.blip_model.vision_proj(image_embeds)
            logits = torch.matmul(projected_image_embeds, text_label_embeds.t())
        return logits