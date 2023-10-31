import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess
from m4c_evaluator import EvalAIAnswerProcessor
from utils.helpers import get_model_name_from_path

def eval_model(args):
    # Model

    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map={"": 0})
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_path, model_type=args.model_type, is_eval=True, device='cuda')
    model_name = get_model_name_from_path(args.model_path)
    with open(args.data_file) as f:
        questions = [json.loads(q) for q in f]
    # questions = np.random.permutation(questions)[:args.subset]
    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    paths = args.image_folder.split('/')
    answers_file = os.path.join("/groups/sernam/adv_llava/results/responses/vqav2/", model_name+args.model_type + '_' + paths[-1]+'_'+os.environ.get('SLURM_JOB_ID', '') + ".json")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    answer_processor = EvalAIAnswerProcessor()
    results = []
    for line in tqdm(questions, total=len(questions)):
        # inputs = processor(images=Image.open(os.path.join(args.image_folder, line['image'])).convert('RGB'), text=args.prompt_format.format(line['text']) , return_tensors="pt").to(device="cuda", dtype=torch.float16)
        file_path = os.path.join(args.image_folder, line['image'])
        if args.image_ext == 'pt':
            file_path = os.path.splitext(file_path)[0] + '.pt'
            line['image'] = torch.load(file_path).unsqueeze(0).cuda().half()
        else:
            line['image'] = vis_processors['eval'](Image.open(file_path).convert('RGB')).unsqueeze(0).cuda().half()
        
        line['text_input'] = args.prompt_format.format(line['text']) 
        with torch.inference_mode():
            result = model.predict_answers(line, inference_method="generate")[0]

        results.append({"question_id": line['question_id'], "answer": answer_processor(result)})

    with open(answers_file, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="blip2_opt")
    parser.add_argument("--model-type", type=str, default="pretrain_opt2.7b") 
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--image_ext", type=str, default="pt")
    parser.add_argument("--data-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--subset", type=int, default=100000)
    parser.add_argument("--prompt_format", type=str, default="Question: {} Short answer:")
    args = parser.parse_args()
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    eval_model(args)
