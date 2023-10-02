import transformers
import json
import torch
import os, re
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import argparse
from transformers import AutoTokenizer
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
import torchvision.transforms as transforms

from tqdm import tqdm

"""
Vanilla
"""


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape((1,3,1,1))
std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape((1,3,1,1))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

criterion = torch.nn.CrossEntropyLoss()

def get_different_class(c_true, classes):
    classes_kept = [c for c in classes if c != c_true]
    return np.random.choice(classes_kept)

def run_LLaVA(args, model, tokenizer, image_tensor):
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
    # print(outputs)
    sentences = outputs.split('\n')
    # sentences = [sentence[3:].strip() for sentence in sentences if sentence]
    parsed_responses = []
    for response in sentences:
        match = re.search(r':\s(.*)', response)
        if match:
            parsed_responses.append(match.group(1))
    # print(parsed_responses)
    return parsed_responses, model.model.image_cls_token

def eval_model(args):

    if args.dataset == 'coco':
        image_list = './coco_test.txt'
        label_list = './coco_label.json'
        # args.set_num = 50
    
    elif args.dataset == 'imagenet':
        image_list = './imagenet_test.txt'
        label_list = './imagenet_label.json'
        args.set_num = 5000
    
    else:
        print("Wrong dataset!")

    f = open(image_list, 'r')
    image_list = f.readlines()
    f.close()

    f = open(label_list, 'r')
    label_names = list(json.load(f).keys())
    f.close()

    label_all = []

    for v in label_names:
        label_all.append("a photo of %s"%v)

    disable_torch_init()

    vision_model = transformers.CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    text_model = transformers.CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    test_transform = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
     
    model_name = os.path.expanduser(args.model_name)
    llava_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llava_model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()

    vision_model.eval()
    text_model.eval()
    llava_model.eval()

    num_image_correct = 0
    num_text_correct = 0
    num_adv_correct = 0
    num_adv_text_correct = 0

    f = open(f"./output/output_{args.dataset}_{args.set}.txt", 'w')

    lower_bound = args.set*args.set_num
    upper_bound = (args.set+1)*args.set_num

    print(f'image_list bounds - Lower bound: {lower_bound}, Upper Bound: {upper_bound}')

    image_list = image_list[args.set*args.set_num:(args.set+1)*args.set_num]

    with torch.no_grad():
        text_labels = tokenizer(label_all, padding=True, return_tensors="pt")['input_ids'].cuda()
        text_label_embeds = text_model(text_labels).text_embeds
        text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

    for i, data in enumerate(tqdm(image_list)):
        image_name, label_name = data.split('|')
        image = Image.open(image_name).convert('RGB')
        image = test_transform(image).cuda().half().unsqueeze(0)
        # image = test_transform(image).cuda().unsqueeze(0)

        ## Text label embedding
        label_name = label_name.split('\n')[0]
        label = label_names.index(label_name)

        with torch.no_grad():
            if args.adv:
                tmp = image_name.split(os.sep)
                if args.dataset=='coco':
                    # tmp[tmp.index('trainval2014')] = 'adv_trainval2014'
                    # tmp[tmp.index('trainval2014')] = 'cutout_trainval2014'
                    # tmp[tmp.index('trainval2014')] = 'cutout2_trainval2014'
                    # tmp[tmp.index('trainval2014')] = 'adv_pgd_trainval2014'
                    # tmp[tmp.index('trainval2014')] = 'adv_fgsm_trainval2014'
                    # tmp[tmp.index('trainval2014')] = 'crop_resize'
                    tmp[tmp.index('trainval2014')] = args.experiment
                elif args.dataset=='imagenet':
                    tmp[tmp.index('val')] = 'adv'

                adv_image_name = os.path.splitext(os.sep + os.path.join(*tmp))[0] + ".pt"
                adv_image = torch.load(os.path.splitext(os.sep + os.path.join(*tmp))[0] + ".pt").cuda()


            text_labels = tokenizer(label_all, padding=True, return_tensors="pt")['input_ids'].cuda()
            text_label_embeds = text_model(text_labels).text_embeds
            text_label_embeds = text_label_embeds / text_label_embeds.norm(p=2, dim=-1, keepdim=True)

            ## LLaVA text embedding
            response, image_cls_token = run_LLaVA(args, llava_model, llava_tokenizer, image)
            
            image_embeds = vision_model.visual_projection(image_cls_token)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_response = tokenizer(response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
            text_response_embeds = text_model(text_response).text_embeds
            text_response_embeds = text_response_embeds / text_response_embeds.norm(p=2, dim=-1, keepdim=True)

            attention_scores = torch.nn.functional.softmax(torch.mm(image_embeds, text_response_embeds.t()) / args.temp, dim=1)
            weighted_sum = torch.mm(attention_scores, text_response_embeds)

            combined_embeds = image_embeds * (1.0-args.scale) + weighted_sum * args.scale
            combined_embeds = combined_embeds / combined_embeds.norm(p=2, dim=-1, keepdim=True)

            ## zero-shot result with image
            logits_per_image = torch.matmul(image_embeds, text_label_embeds.t())

            ## zero-shot result with llava response
            # logits_per_text = torch.matmul(text_response_embeds, text_label_embeds.t()).mean(dim=0)
            logits_per_text = torch.matmul(combined_embeds, text_label_embeds.t())
            
            if logits_per_image.argmax(dim=-1).item()==label:
                num_image_correct += 1

            if logits_per_text.argmax(dim=-1).item()==label:
                num_text_correct += 1

            if args.adv:
                 ## Adv LLaVA text embedding
                adv_response, adv_image_cls_token  = run_LLaVA(args, llava_model, llava_tokenizer, adv_image)

                adv_image_embeds = vision_model.visual_projection(adv_image_cls_token)
                adv_image_embeds = adv_image_embeds / adv_image_embeds.norm(p=2, dim=-1, keepdim=True)
                adv_text_response = tokenizer(adv_response, padding=True, truncation=True, return_tensors="pt")['input_ids'].cuda()
                adv_text_response_embeds = text_model(adv_text_response).text_embeds
                adv_text_response_embeds = adv_text_response_embeds / adv_text_response_embeds.norm(p=2, dim=-1, keepdim=True)
                
                adv_attention_scores = torch.nn.functional.softmax(torch.mm(adv_image_embeds, adv_text_response_embeds.t()) / args.temp, dim=1)
                adv_weighted_sum = torch.mm(adv_attention_scores, adv_text_response_embeds)

                # adv_combined_embeds = adv_weighted_sum
                adv_combined_embeds = adv_image_embeds * (1.0-args.adv_scale) + adv_weighted_sum * args.adv_scale
                adv_combined_embeds = adv_combined_embeds / adv_combined_embeds.norm(p=2, dim=-1, keepdim=True)

                ## zero-shot result with adv image
                adv_logits_per_image = torch.matmul(adv_image_embeds, text_label_embeds.t())

                ## zero-shot result with adv llava response
                adv_logits_per_text = torch.matmul(adv_combined_embeds, text_label_embeds.t()).mean(dim=0)

                if adv_logits_per_image.argmax(dim=-1).item()==label:
                    num_adv_correct += 1

                if adv_logits_per_text.argmax(dim=-1).item()==label:
                    num_adv_text_correct += 1

                if (i+1) % 10 == 0:
                    print(i + args.set*args.set_num, "Image: %.4f"%(num_image_correct/(i+1)), "Text: %.4f"%(num_text_correct/(i+1)), "adv Acc: %.4f"%(num_adv_correct/(i+1)), "adv Text Acc: %.4f"%(num_adv_text_correct/(i+1)), flush=True)
            else:
                if (i+1) % 10 == 0:
                    print(i + args.set*args.set_num, "Image: %.4f"%(num_image_correct/(i+1)), "Text: %.4f"%(num_text_correct/(i+1)), flush=True)
            if logits_per_image.argmax(dim=-1).item() != logits_per_text.argmax(dim=-1).item():
                line2write = "%s|%s|%s|%s|%s"%(image_name, label_name, label_names[logits_per_image.argmax(dim=-1).item()], label_names[logits_per_text.argmax(dim=-1).item()], response)
                print(line2write, flush=True)
                f.write(line2write+'\n')

    # print(i + args.set*args.set_num, "Image: %.4f"%(num_image_correct/(i+1)), "Text: %.4f"%(num_text_correct/(i+1)), "adv Acc: %.4f"%(num_adv_correct/(i+1)), "adv Text Acc: %.4f"%(num_adv_text_correct/(i+1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./LLAMA-on-LLaVA")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--image_size", type=float, default=224)
    parser.add_argument("--set", type=int, required=True)
    parser.add_argument("--set_num", type=int, default=1000)
    parser.add_argument("--llava_temp", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--adv_scale", type=float, default=0.95)
    parser.add_argument("--adv", type=boolean_string, default='True')
    parser.add_argument("--query", type=str, default="Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None, required=True)
    args = parser.parse_args()

    eval_model(args)

