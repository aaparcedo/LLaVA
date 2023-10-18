import torch
import os, re
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from llava.model import *
from attacks.helpers import *
from llava.model.utils import KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# unwanted_words = ['sure', 'okay', 'yes', 'of course', 'yeah', 'no problem']

def run_LLaVA(args, llava_model, llava_tokenizer, image_tensor, text=None, vqa=False):

    llava_model.eval()

    # Model
    mm_use_im_start_end = getattr(llava_model.config, "mm_use_im_start_end", False)
    llava_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        llava_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    if 'v1.5' in args.model_name:
        vision_tower = llava_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to('cuda', dtype=torch.float16)
    else:
        vision_tower = llava_model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        llava_model.get_model().vision_tower[0] = vision_tower

    vision_config = vision_tower.config

    vision_config.im_patch_token = llava_tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = llava_tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    qs = args.query if not text else text
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if not vqa:
        conv_mode = "multimodal"
        
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = llava_tokenizer([prompt])

    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, llava_tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = llava_model.generate(
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
    outputs = llava_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    sentences = outputs.split('\n')
    parsed_responses = []

    if len(sentences) == 1:
        return sentences[0], llava_model.model.image_cls_token

    for response in sentences:
        
        if 'Content: ' in sentences[0]:
            match = re.search(r':\s(.*)', response)
        else:
            match = re.search(r'\d\.(.*)', response)
        if match:
            parsed_responses.append(match.group(1))
    return parsed_responses, llava_model.model.image_cls_token


    


if __name__ == '__main__':
    llava_tokenizer = AutoTokenizer.from_pretrained('/groups/sernam/ckpts/LLAMA-on-LLaVA')
    llava_model = LlavaLlamaForCausalLM.from_pretrained('/groups/sernam/ckpts/LLAMA-on-LLaVA', low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
    img = Image.open('/groups/sernam/datasets/imagenet/val/ILSVRC2012_val_00000001.JPEG')
    run_LLaVA(llava_model, llava_tokenizer, image_processor().cuda())