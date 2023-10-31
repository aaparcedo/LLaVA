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
from lavis.models import load_model_and_preprocess, Blip2ITM, load_preprocess
from omegaconf import OmegaConf
from lavis.common.registry import registry
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
        
    def preprocess_image(self, image):
        return self.processor(images=image, return_tensors="pt")['pixel_values'][0]

    def forward(self, images, text_label_embeds, is_attack=False, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        """

        def inner():
            image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
                    ## zero-shot result with clip
            logits = torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label
            return logits
        
        if is_attack:
            logits = inner()
        else:
            with torch.inference_mode():
                logits = inner()
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
    
    def preprocess_image(self, image):
        return self.processor(images=image, return_tensors="pt")['pixel_values'][0]

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
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-13b", None, 'llava-v1.5-13b')
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=torch.float16).cuda()

        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
    
    def preprocess_image(self, image):
        return self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        # return process_images([image], self.image_processor, self.model.config)[0]
    
    def generate(self, input_ids, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # make it a batch
        with torch.inference_mode():
            response = run_llava2(self.args, model=self.model, 
                            model_path=self.model_path,
                            tokenizer=self.tokenizer, 
                            image_tensor=image, input_ids=input_ids)
        return response

    def forward(self, images, text_label_embeds, is_attack=False, return_response=False, input_ids=None, text=None, **kwargs):
        """
            Generate responses and logits for image-caption/label retrieval via LLAVA. 
            *** In this case, the input_ids has to be the same for all images. (e.g. a standard query: "describe the image") ***
        """
        if is_attack:
            image_embeds = F.normalize(self.vision_encoder(images).image_embeds, p=2., dim=-1)
            if text is not None:
                input_ids = self.text_tokenizer(text=text, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                text_embeds = F.normalize(self.text_encoder(input_ids).text_embeds, p=2., dim=-1)
                return torch.matmul(image_embeds, text_embeds.t())
            # for attack, we only attack the clip part of llava
            else:
                return torch.matmul(image_embeds, text_label_embeds.t()) # B, n_label

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


class BLIP2ITM(nn.Module):
    
    def __init__(self, args) -> None:
        """
        *** Only support batch_size=1
        """
        super().__init__()
        assert args.batch_size == 1, "BLIP2 currently only support batch_size=1"
        self.args = args
        self.itm_model, self.vis_processors, self.text_processors = load_model_and_preprocess(name="blip2_image_text_matching", model_type="pretrain", is_eval=True, device='cuda')
        for param in self.itm_model.parameters():
            param.requires_grad = True
        
    def preprocess_image(self, image):
        return self.vis_processors['eval'](image)

    def forward(self, images, text=None, is_attack=False, match_head='itm', sentence_labels=None, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        if text is not None, then to return contrastive loss
        otherwise, return logits for classification based on itm or itc
        """
        def inner():
            if text:
                return self.itm_model({"image": images, "text_input": text}, match_head='itc')
            else:
                logits = []
                for label in sentence_labels:
                    if match_head == 'itm':
                        logits.append(self.itm_model({'image': images, 'text_input': label}, match_head='itm')[:,1])
                    else:
                        logits.append(self.itm_model({'image': images, 'text_input': label}, match_head='itc'))
                    
                logits = torch.stack(logits, dim=0)
                return logits
        if not is_attack:
            with torch.inference_mode():
                return inner()
        else:
            return inner()

class BLIP2CL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True)
        self.args = args
        self.tokenizer = self.model.tokenizer
    
    def preprocess_image(self, image):
        return self.vis_processors['eval'](image)
    
    def forward(self, images, text_label_embeds=None, texts=None, is_attack=False, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        if text is not None, then to return contrastive loss
        otherwise, return logits for classification based on itm or itc
        """
        def inner():
            image_feat = self.model.extract_features({"image": images}, mode='image').image_embeds_proj
            if texts:
            # image_embeds_proj and text_embeds_proj are normalized already
                text_feat = self.model.extract_features({"text_input": texts}, mode='text').text_embeds_proj
            else:
                text_feat = text_label_embeds
            return (image_feat @ text_feat[:,0,:].t()).max(1)[0]
                
        if not is_attack:
            with torch.inference_mode():
                return inner()
        else:
            return inner()
    @torch.no_grad()
    def encode_images(self, images):
        # B dim
        return self.model.extract_features({"image": images}, mode='image').image_embeds_proj
    @torch.no_grad()
    def encode_texts(self, texts):
        # B # query_tokens dim
        return self.model.extract_features({"text_input": texts}, mode='text').text_embeds_proj


class InstructBLIP(nn.Module):
    
    def __init__(self, args) -> None:
        """
        *** Only support batch_size=1
        """
        super().__init__()
        self.args = args
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type=args.model_type, is_eval=True)
        
        self.text_tokenizer = clip_text_tokenizer
        self.text_encoder = clip_text_encoder
    
    def preprocess_image(self, image):
        return self.vis_processors['eval'](image)

    def generate(self, text, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # make it a batch
        with torch.inference_mode():
            return self.model.predict_answers({"image": image, 'text_input': text}, inference_method="generate")[0]
    
        
    def forward(self, images, text_label_embeds, text=None, **kwargs):
        """
        Forward pass to generate logits for image-caption/label retrieval
        if text is not None, then to return contrastive loss
        otherwise, return logits for classification based on itm or itc
        """

        with torch.inference_mode():
            response = self.generate(text, images)
            text_response = self.text_tokenizer(text=response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_response_embed = F.normalize(self.text_encoder(text_response).text_embeds, p=2., dim=-1)
            
            logits = torch.matmul(text_response_embed, text_label_embeds.t())
        
        return logits


    def encode_texts(self, texts):
        with torch.no_grad():
            text_label_embeds = self.text_encoder(self.model.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()).text_embeds
        
        return F.normalize(text_label_embeds, dim=-1)


def get_model(args):
    if 'llava-v1.5' in args.model_path.lower():
        model = LLaVA2(args)
    elif 'llava' in args.model_path.lower():
        model = LLaVA(args)
    elif 'clip' in args.model_path.lower():
        model = CLIP(args)
    elif args.model_path.lower() == 'blip2_image_text_matching':
        model = BLIP2CL(args)
    elif 'instruct' in args.model_path.lower():
        model = InstructBLIP(args)
    
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