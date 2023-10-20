import os
from transformers import AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, \
                         BlipModel, Blip2ForConditionalGeneration, CLIPModel, BlipForImageTextRetrieval, AutoProcessor, \
                        BlipForQuestionAnswering, AutoProcessor, Blip2VisionModel
                        
from llava.model import *
import torch
from PIL import Image
import requests
import torch.nn.functional as F
import torch.nn as nn
from run_llava import run_LLaVA, run_llava2
from llava.modelv2.builder import load_pretrained_model
from m4c_evaluator import EvalAIAnswerProcessor
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# For encoding labels/captions and the generated response, we use the same CLIP text encoder
clip_text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
clip_text_encoder = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()
clip_processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')

class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(args.model_path, torch_dtype=torch.float16).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(args.model_path)

        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
        

    def forward(self, images, text_label_embeds, is_attack=False, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """

        image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
                ## zero-shot result with clip
        logits = torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label
        return logits
    
class LLaVA(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("/groups/sernam/ckpts/LLAMA-on-LLaVA")
        self.model = LlavaLlamaForCausalLM.from_pretrained("/groups/sernam/ckpts/LLAMA-on-LLaVA", low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda().eval()
        self.processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()

        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder

    def forward(self, images, text_label_embeds, is_attack=False, return_response=False, input_ids=None, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        Note: We don't attack the LLaVA LLM part, only the CLIP part
        """

        if is_attack:
            # for attack, we only attack the clip part of llava
            image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
                    ## zero-shot result with clip
            logits = torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label
            return logits

        with torch.inference_mode():
            responses, text_response_embeds = [], []
            for image in images:
                response, _ = run_LLaVA(self.args, self.model, self.tokenizer, image.unsqueeze(0))
                responses.append(response)
                text_response = self.text_tokenizer(text=response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_response_embed = F.normalize(self.text_encoder(text_response).text_embeds, p=2., dim=-1)
                text_response_embeds.append(text_response_embed)

            text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768

            # else:
            logits = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=1)
        if return_response:
            return logits, responses
        else:
            return logits


class LLaVA2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_path = args.model_path
        self.processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-13b", None, self.model_name)
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=torch.float16).cuda()

        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
    
    def generate(self, input_ids, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # make it a batch
        with torch.inference_mode():
            response = run_llava2(self.args, model=self.model, 
                            model_path=self.model_path,
                            tokenizer=self.tokenizer, 
                            image_tensor=image, input_ids=input_ids)
        return response

    def forward(self, images, text_label_embeds, is_attack=False, return_response=False, input_ids=None):
        """
            Generate responses and logits for image-caption/label retrieval via LLAVA. 
            *** In this case, the input_ids has to be the same for all images. (e.g. a standard query: "describe the image") ***
        """
        if is_attack:
            # for attack, we only attack the clip part of llava
            image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
            logits = torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label
            return logits

        with torch.inference_mode():
            responses, text_response_embeds = [], []
            for image in images:
                response = self.generate(input_ids, image)
                # add the below line for classification -- llava1.5 generate a one-word description of the image
                response = f"a photo of {response}."
                responses.append(response)
                text_response = self.text_tokenizer(text=response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_response_embed = F.normalize(self.text_encoder(text_response).text_embeds, p=2., dim=-1)
                text_response_embeds.append(text_response_embed)

            text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768
            logits = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=1)
        if return_response:
            return logits, responses
        else:
            return logits


class BLIP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        
        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder

        self.answer_processor = EvalAIAnswerProcessor()

    def generate(self, input_ids, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # make it a batch
        with torch.inference_mode():
            generated_ids = self.model.generate(pixel_values=image, 
                                                input_ids=input_ids, 
                                                max_new_tokens=self.args.max_new_tokens, 
                                                min_length=self.args.min_len,
                                                num_beams=self.args.num_beams) # add a conditional query to continue from the query
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return response

        
    def forward(self, images, text_label_embeds, is_attack=False, input_ids=None):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """

        if is_attack:
            image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
            logits = torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label
            return logits
        
        text_response_embeds = []
        for image in images:
            generated_text = self.generate(input_ids, image)
            text_response = self.text_tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_response_embed = F.normalize(self.text_encoder(text_response).text_embeds, p=2., dim=-1)
            text_response_embeds.append(text_response_embed)

        text_response_embeds = torch.stack(text_response_embeds, dim=0) # B, n_sentence, 768
        logits = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=1)
        return logits





def get_model(args):
    if 'llava-v1.5' in args.model_path.lower():
        model = LLaVA2(args)
    elif 'llava' in args.model_path.lower():
        model = LLaVA(args)
    elif 'clip' in args.model_path.lower():
        model = CLIP(args)
    elif 'blip' in args.model_path.lower():
        model = BLIP(args)
    
    return model

if __name__ == '__main__':


    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 ).cuda().eval()
    image = "/groups/sernam/datasets/coco/val2014/COCO_val2014_000000000192.jpg"
    image = Image.open(image).convert('RGB')
    prompt = "What is the main object in this image? Answer: "
    inputs = processor(images=image, text=prompt, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=1)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)