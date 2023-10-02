import transformers
import json
import torch
import os, re
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import argparse
from transformers import AutoTokenizer
from attacks.generate_adv_samples import generate_one_adv_sample
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
import torchvision.transforms as transforms
from attacks.helpers import *
from tqdm import tqdm
import torch.nn.functional as F

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

@torch.inference_mode()
def run_LLaVA(args, image_tensor):

    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    model.eval()

    # Model
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower

    vision_config = vision_tower.config

    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    qs = args.query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    conv_mode = "multimodal"
    
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.llava_temp,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    sentences = outputs.split('\n')
    parsed_responses = []
    for response in sentences:
        match = re.search(r':\s(.*)', response)
        if match:
            parsed_responses.append(match.group(1))
    return parsed_responses, model.model.image_cls_token

@torch.inference_mode()
def main(args):

    if args.dataset == 'coco':
        image_list = './coco_test.txt'
        label_list = './coco_test.json'
        # args.subset = 50
    
    elif args.dataset == 'imagenet':
        image_list = './imagenet_test.txt'
        label_list = './imagenet_label.json'
        # args.subset = 5000
    
    else:
        print("Wrong dataset!")

    f = open(image_list, 'r')
    image_list = f.readlines()
    f.close()

    label_names = set()
    for line in image_list:
        label = line.split('|')[-1].split('\n')[0]
        label_names.add(label)
    label_names = list(label_names)

    image_list = np.random.permutation(image_list)[:args.subset]

    # f = open(label_list, 'r')
    # label_names = list(json.load(f).keys())
    # f.close()

    disable_torch_init()

    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    test_transform = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    vision_model.eval()
    text_model.eval()

    num_image_correct = 0
    num_text_correct = 0
    text_labels = ["a photo of %s"%v for v in label_names]
    text_labels = tokenizer(text_labels, padding=True, return_tensors="pt")['input_ids'].cuda()
    text_label_embeds = text_model(text_labels).text_embeds
    text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

    # check if the corresponding adv image folder already exist
    adv_img_folder = f"../datasets/{args.dataset}/{args.attack_name}_eps{args.eps}_lr{args.lr}_steps{args.nb_iter}_norm{args.norm}"
    if os.path.exists():
        adv_exist = True
    else:
        adv_exist = False

    for i, path in enumerate(tqdm(image_list)):
        # load & transform image
        image_path, label_name = path.split('|')
        if not args.adv:
            image = Image.open(image_path).convert('RGB')
            image = test_transform(image).cuda().half().unsqueeze(0)
        else:
            if adv_exist:
                image = Image.open(os.path.join(adv_img_folder, os.path.basename(image_path))).convert('RGB')
                image = test_transform(image).cuda().half().unsqueeze(0)
            else:
                image = Image.open(image_path).convert('RGB')
                image = test_transform(image).cuda().half().unsqueeze(0)
                image = generate_one_adv_sample(image, 
                                args.attack_name, 
                                text_label_embeds, 
                                vision_model, 
                                save_image=args.save_image, 
                                image_name=os.path.basename(image_path), 
                                eps=args.eps, 
                                lr=args.lr, 
                                nb_iter=args.nb_iter, 
                                norm=args.norm)
        
        ## Text label embedding
        label_name = label_name.split('\n')[0]
        label = label_names.index(label_name)

        ## LLaVA text embedding
        response, image_cls_token = run_LLaVA(args, image)
        
        image_embeds = F.normalize(vision_model.visual_projection(image_cls_token), p=2., dim=-1)
        
        text_response = tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
        text_response_embeds = F.normalize(text_model(text_response).text_embeds, p=2., dim=-1)

        attention_scores = torch.nn.functional.softmax(torch.mm(image_embeds, text_response_embeds.t()) / args.temp, dim=1)
        weighted_sum = torch.mm(attention_scores, text_response_embeds)
        combined_embeds = F.normalize(image_embeds * (1.0-args.scale) + weighted_sum * args.scale, p=2., dim=-1)

        ## zero-shot result with image
        logits_per_image = torch.matmul(image_embeds, text_label_embeds.t())

        ## zero-shot result with llava response
        # logits_per_text = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=0)
        logits_per_text = torch.matmul(combined_embeds, text_label_embeds.t())
        
        if logits_per_image.argmax(dim=-1).item()==label:
            num_image_correct += 1

        if logits_per_text.argmax(dim=-1).item()==label:
            num_text_correct += 1

    with open("./outputs/results.txt", "a") as f:
        f.write('='*50+'\n')
        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("==> Image Acc: {}    Text Acc: {}".format(num_image_correct/len(image_list), num_text_correct/len(image_list)))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./ckpts/LLAMA-on-LLaVA")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--image_size", type=float, default=224)
    parser.add_argument("--attack_name", type=str, default="pgd")
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--adv", type=boolean_string, default='True')
    parser.add_argument("--save_image", type=boolean_string, required=False, default='False')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--nb_iter", type=int, default=30)
    parser.add_argument("--norm", type=float, default=np.inf)
    parser.add_argument("--query", type=str, default="Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}")
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    if not args.adv:
        args.lr = None
        args.nb_iter = None
        args.norm = None

    main(args)