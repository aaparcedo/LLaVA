import transformers
import json
import torch
import os
import sys
import argparse

from PIL import Image
from transformers import AutoTokenizer
from llava.utils import disable_torch_init
from llava.model import *
from utils.func import *

import torchvision.transforms as transforms


"""
Descriptors attack
"""

def eval_model(args):
    
    ## datasets
    # coco: 28000, 80 classes
    # imagenet: 50000, 1000 classes
    # cub: 11788, 200 clases
    # eurosat: 27000, 10 classes

    image_list = f'{args.dataset}_test.txt'
    descriptor_list = f'descriptors/descriptors_{args.dataset}.json'

    f = open(image_list, 'r')
    image_list = f.readlines()
    f.close()

    f = open(descriptor_list, 'r')
    descriptors = json.load(f)
    label_names = list(descriptors.keys())
    f.close()

    disable_torch_init()

    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    test_transform = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
     
    model_name = os.path.expanduser(args.model_name)
    llava_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llava_model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()

    vision_model.eval()
    text_model.eval()
    llava_model.eval()
    
    num_image_correct = 0
    num_combined_correct = 0

    adv_num_image_correct = 0
    adv_num_combined_correct = 0

    # set_num = len(image_list) // 8
    set_num = args.set_num
    # image_list = image_list[args.set*set_num:(args.set+1)*set_num]


    with torch.no_grad():
        text_label_embeds = []

        for label in label_names:
            examples = descriptors[label]
            sentences = []
            for example in examples:
                sentence = f"{label} {make_descriptor_sentence(example)}"
                sentences.append(sentence)
            # print(f'sentences: {sentences}', flush=True) # tench which is a freshwater fish'... (4 more)
            text_descriptor = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            # print(f'text descriptor shape: {text_descriptor.shape}', flush=True) # torch.Size([7, 15])
            # print(f'text descriptor: {text_descriptor}', flush=True) # sentence tokens
            text_descriptor_embeds = text_model(text_descriptor).text_embeds
            # print(f'text descriptor embeds shape: {text_descriptor_embeds.shape}', flush=True) # torch.Size([6, 768])
            text_descriptor_embeds = text_descriptor_embeds / text_descriptor_embeds.norm(p=2, dim=-1, keepdim=True)
            text_label_embeds.append(text_descriptor_embeds)
            
    # print(f'text label embedding len: {len(text_label_embeds)}', flush=True) # 1000
            
    # print(f"label name: {label_names[0]}", flush=True) # 'tench'
    # print(f'descriptor: {descriptors[label_names[0]]}', flush=True) # 'fresh water fish'... (4 more)
    
    # print(f'type text label embeds: {type(text_label_embeds)}', flush=True) # list
    # print(f'type text label embeds[0]: {type(text_label_embeds[0])}', flush=True) # torch.Tensor
    
    for i, data in enumerate(image_list):
        image_name, label_name = data.split('|')
        image = Image.open(image_name.replace('/imagenet/', '/imagenet/val/')).convert('RGB')
        image = test_transform(image).cuda().half().unsqueeze(0)

        label_name = label_name.split('\n')[0]
        label = label_names.index(label_name)

        with torch.no_grad():
            ## Run on original image
            llava_response, image_cls_token = run_LLaVA(args, llava_model, llava_tokenizer, image)
            image_embeds = vision_model.visual_projection(image_cls_token)
            text_response = tokenizer(llava_response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_response_embeds = text_model(text_response).text_embeds

            logits_image, logits_combined = classify_with_descriptors(text_response_embeds, image_embeds, text_label_embeds, temp=args.temp, scale=args.scale)            
            
            if logits_image.argmax(dim=-1).item()==label:
                num_image_correct += 1

            if logits_combined.argmax(dim=-1).item()==label:
                num_combined_correct += 1

            if args.adv:
                adv_image_path = image_name.replace("/imagenet/", "/imagenet/pgd/descriptors/").replace('JPEG', 'pt')
                adv_image = torch.load(adv_image_path).cuda()

                 ## Adv LLaVA text embedding and cls token
                adv_response, adv_image_cls_token  = run_LLaVA(args, llava_model, llava_tokenizer, adv_image)

                adv_image_embeds = vision_model.visual_projection(adv_image_cls_token)
                adv_text_response = tokenizer(adv_response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                adv_text_response_embeds = text_model(adv_text_response).text_embeds

                adv_logits_per_image, adv_logits_per_text = classify_with_descriptors(adv_text_response_embeds, adv_image_embeds, text_label_embeds, temp=args.temp, scale=args.adv_scale)            

                if adv_logits_per_image.argmax(dim=-1).item()==label:
                    adv_num_image_correct += 1

                if adv_logits_per_text.argmax(dim=-1).item()==label:
                    adv_num_combined_correct += 1

        if (i+1) % 10 == 0:
                print((i+1) + args.set*set_num, "CLIP: %.4f"%(num_image_correct/(i+1)), "LLaVA: %.4f"%(num_combined_correct/(i+1)), "adv CLIP: %.4f"%(adv_num_image_correct/(i+1)), "adv CLIP + LLaVA: %.4f"%(adv_num_combined_correct/(i+1)), flush=True)

        line2write = "image: %s| true label: %s| prediction clip: %s| prediction llava %s| llava response: %s"%(image_name, label_name, label_names[logits_image.argmax(dim=-1).item()], label_names[logits_combined.argmax(dim=-1).item()], llava_response)
        print(line2write, flush=True)
        # f.write(line2write+'\n')

    print((i+1) + args.set*set_num, "Image: %.4f"%(num_image_correct/(i+1)), "Text: %.4f"%(num_combined_correct/(i+1)), flush=True)

if __name__ == "__main__":

    arg_index = sys.argv.index('--set') + 1 if '--set' in sys.argv else None
    set_value = sys.argv[arg_index]
    arg_index_dataset = sys.argv.index('--dataset') + 1 if '--dataset' in sys.argv else None
    dataset_value = sys.argv[arg_index_dataset] if arg_index_dataset else "imagenet"
    arg_index_scale = sys.argv.index('--scale') + 1 if '--scale' in sys.argv else None
    scale_value = sys.argv[arg_index_scale] if arg_index_scale else "0.1"

    # log_filename = f"test_logs/gpt_{dataset_value}_{scale_value}_{set_value}.log"
    log_filename = f"test_logs/gpt_adv_{dataset_value}_{scale_value}_{set_value}.log"

    sys.stdout = open(log_filename, 'w')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./ckpts/LLAMA-on-LLaVA")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--image_size", type=float, default=224)
    parser.add_argument("--set", type=int, default=0)
    parser.add_argument("--set_num", type=int, default=6250)
    parser.add_argument("--llava_temp", type=float, default=0.8)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--adv_scale", type=float, default=0.95)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--query", type=str, default="For fine-grained image classification, first give a general description of the image. Then, point out the small details and unique features of this image.")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--adv", type=boolean_string, default='True')
    args = parser.parse_args()

    eval_model(args)

    
"""
python classification_gpt.py --set 0 --set_num 50000 
"""

