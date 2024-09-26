from transformers import AutoTokenizer, CLIPProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
from llava.model import *
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from mids.selector import make_decision
from utils.file_utils import *
from utils.llava_utils import *
from utils.mids_utils import *
from tqdm import tqdm
import argparse
import threading
import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_llava(model_path, device_id):
    kwargs = {"device_map": device_id, 'torch_dtype': torch.float16}

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, assign = True)
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    return model, image_processor, tokenizer

def load_mids(model_path, device_id):
    print(model_path)
    from mids.mids_arch import MIDS
    model = MIDS()

    model_state_dict = model.state_dict()

    finetuned_state_dict = torch.load(model_path)
    finetuned_state_dict = {key.replace('module.', ''): value for key, value in finetuned_state_dict.items()}

    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)

    return model.to(dtype=torch.float32, device=torch.device(f'cuda:{device_id}'))

def get_llava_prompt(model, qs, conv_mode):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def get_llava_answer(model, tokenizer, image_processor, image, prompt_list,
                     temperature, top_p, num_beams, max_new_tokens,
                     per_sample_generate_num, conv_mode='v1'):
    image_size = [image.size]
    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    outputs = []

    # prepare prompts
    condition_prompt = 'This is a _ human face. What evidence do you have?'
    prompts = []
    prompts.append(random.choice(prompt_list))

    with torch.inference_mode():
        for i in range(per_sample_generate_num):
            qs = prompts[i]
            prompt = get_llava_prompt(model, qs, conv_mode)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_size,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
                pad_token_id=tokenizer.pad_token_id,
            )

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if i == 0:
                response_json, _ = decode_response(output)
                if response_json['Analysis result'].lower() == 'real':
                    prompts.append(condition_prompt.replace('_', 'fake'))
                    prompts.append(condition_prompt.replace('_', 'real'))
                else:
                    prompts.append(condition_prompt.replace('_', 'real'))
                    prompts.append(condition_prompt.replace('_', 'fake'))
            outputs.append(output)

        return outputs
    

def eval(args):
    gt = read_json(args.gt)

    llava_groups = args.llava_groups
    mids_path = args.mids_path
    mask_answer = 'nomask' not in mids_path
    print('mask_answer:', mask_answer)

    res_list = []
    y_true = []
    y_pred = []
    y_scores = []

    decision_count = {'reason': [0, 1e-8], 'c_same':[0, 1e-8], 'c_opposite': [0, 1e-8]}
    difficulty_count = {'easy': [0, 1e-8], 'hard': [0, 1e-8]}

    # load model
    device_idx = 0
    print(args.generate_num)
    print(args.benchmark)
    if args.generate_num['mistral'] > 0:
        mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(llava_groups["mistral"], device_idx)
        if args.generate_num['phi'] > 0: device_idx += 1
    if args.generate_num['phi'] > 0:
        phi_model, phi_image_processor, phi_tokenizer = load_llava(llava_groups["phi"], device_idx)

    t5_tokenizer = AutoTokenizer.from_pretrained('models/t5-base', use_fast=False, legacy=False)
    clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14-336")
    mids = load_mids(mids_path, device_idx)
    mids.eval()

    device = torch.device(f'cuda:{device_idx}')
    def run_mistral(image, answer_holder):
        with torch.no_grad():
            answers = get_llava_answer(mistral_model, mistral_tokenizer, mistral_image_processor,
                                    image, args.prompt_list, args.temperature, args.top_p, args.num_beams,
                                    args.max_new_tokens, args.generate_num['mistral'], 'v1')
            answer_holder.extend(answers)

    def run_phi(image, answer_holder):
        with torch.no_grad():
            answers = get_llava_answer(phi_model, phi_tokenizer, phi_image_processor,
                                    image, args.prompt_list, args.temperature, args.top_p, args.num_beams,
                                    args.max_new_tokens, args.generate_num['phi'], 'v1')
            answer_holder.extend(answers)
    processed_ids = set()
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                processed_ids.add(data.get('id'))
    
    processed_num = 0
    test_images = list(gt.keys())
    if args.eval_samples_num > 0:
        test_images = test_images[:args.eval_samples_num]
        
    for imagename in tqdm(test_images):
        image = load_image(os.path.join(args.image_dir, imagename))
        threads = []

        mistral_answer_holder = []
        phi_answer_holder = []
        llama_answer_holder = []

        if args.generate_num['mistral'] > 0:
            mistral_thread = threading.Thread(target=run_mistral, args=(image, mistral_answer_holder))
            threads.append(mistral_thread)
            mistral_thread.start()
        if args.generate_num['phi'] > 0:
            phi_thread = threading.Thread(target=run_phi, args=(image, phi_answer_holder))
            threads.append(phi_thread)
            phi_thread.start()
        
        # wait all threads complete
        for thread in threads:
            thread.join()
        
        answers = mistral_answer_holder + phi_answer_holder + llama_answer_holder
        scores = []

        # mask answers
        answers_result = []
        processed_answers = []
        for answer in answers:
            answer, answer_res = mask_result(answer) if mask_answer else get_result(answer)
            answers_result.append(answer_res)
            processed_answers.append(answer)
        
        # easy or hard
        if len(set(answers_result)) == 1:
            qs_difficulty = 'easy'
        else:
            qs_difficulty = 'hard'
        difficulty_count[qs_difficulty][1] += 1
            
        if len(answers) == 3:
            N=1;M=1
        elif len(answers) == 2:
            N=0;M=1
        elif len(answers) == 1:
            N=0;M=0
        
        # print(answers)
            
        # mids
        with torch.inference_mode():
            input_image = clip_processor(images=image, return_tensors='pt')['pixel_values']
            answer_ids = t5_tokenizer(processed_answers, return_tensors="pt", padding="longest", max_length=t5_tokenizer.model_max_length, truncation=True)
            logits = mids(answer_ids.to(device), input_image.to(device), None, 1, N,M)['logits']
            scores = F.softmax(logits, dim=2).squeeze(0)
            best_answer_idx, pred, match_score, forgery_score = make_decision(answers_result, scores)

        if args.benchmark == 'multiattack':
            gt_cls = 0 if gt[imagename][0].lower()=='real' else 1
        else:
            gt_cls = 0 if gt[imagename].lower()=='real' else 1

        if best_answer_idx == 0:
            decision_count['reason'][1] += 1
            if gt_cls == pred: 
                decision_count['reason'][0] += 1
                difficulty_count[qs_difficulty][0] += 1
        elif best_answer_idx == 1:
            decision_count['c_same'][1] += 1
            if gt_cls == pred: 
                decision_count['c_same'][0] += 1
                difficulty_count[qs_difficulty][0] += 1
        else:
            decision_count['c_opposite'][1] += 1
            if gt_cls == pred: 
                decision_count['c_opposite'][0] += 1
                difficulty_count[qs_difficulty][0] += 1

        y_pred.append(pred)
        y_true.append(gt_cls)
        y_scores.append(forgery_score)
        res_list.append({
            'id': imagename,
            'difficulty': qs_difficulty,
            'correct': gt_cls == pred,
            'match score': round(match_score, 4),
            'content': answers[best_answer_idx],
        })

        processed_num += 1
        if processed_num % 10 == 0:
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            conf_matrix = confusion_matrix(y_true, y_pred)
            tpr = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            tnr = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

            res_list.append({
                'acc': round(acc, 4),
                'auc': round(auc, 4),
                'ap': round(ap, 4),
                'tpr': round(tpr, 4),
                'tnr': round(tnr, 4),
                'reason': round(decision_count['reason'][0] / decision_count['reason'][1], 4),
                'c_same': round(decision_count['c_same'][0] / decision_count['c_same'][1], 4),
                'c_opposite': round(decision_count['c_opposite'][0] / decision_count['c_opposite'][1], 4),
                'easy_num': int(difficulty_count['easy'][1]),
                'hard_num': int(difficulty_count['hard'][1]),
                'easy_acc': round(difficulty_count['easy'][0] / difficulty_count['easy'][1], 4),
                'hard_acc': round(difficulty_count['hard'][0] / difficulty_count['hard'][1], 4)
            })

            write_jsonl(args.save_path, res_list)

    # complete
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tpr = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    tnr = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    res_list.append({
        'acc': round(acc, 4),
        'auc': round(auc, 4),
        'ap': round(ap, 4),
        'tpr': round(tpr, 4),
        'tnr': round(tnr, 4),
        'reason': round(decision_count['reason'][0] / decision_count['reason'][1], 4),
        'c_same': round(decision_count['c_same'][0] / decision_count['c_same'][1], 4),
        'c_opposite': round(decision_count['c_opposite'][0] / decision_count['c_opposite'][1], 4),
        'easy_num': int(difficulty_count['easy'][1]),
        'hard_num': int(difficulty_count['hard'][1]),
        'easy_acc': round(difficulty_count['easy'][0] / difficulty_count['easy'][1], 4),
        'hard_acc': round(difficulty_count['hard'][0] / difficulty_count['hard'][1], 4)
    })

    write_jsonl(args.save_path, res_list)



if __name__ == "__main__":
    """
        CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark pgc --model mistral --generate_num 3 --eval_num -1
    """
    # args
    parser = argparse.ArgumentParser(description="ffaa")
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, default='mistral')
    parser.add_argument('--generate_num', type=int, required=True, default=1)
    parser.add_argument('--eval_num', type=int, default=-1)

    parser_args = parser.parse_args()
    benchmark = parser_args.benchmark
    modelname = parser_args.model
    generate_num = parser_args.generate_num
    eval_num = parser_args.eval_num

    # prompt_list
    prompt_list = read_txt_file("playground/prompts.txt")

    # model path
    llava_groups = {
        'mistral': "checkpoints/ffaa-mistral-7b",
        'phi': "checkpoints/ffaa-phi-3-mini",
    }
    epoch = 2
    mids_path = f"checkpoints/mids/mids_best/{epoch}.pth"
    
    genarate_num_dict = {
        'mistral': 0,
        'phi': 0,
    }

    if modelname != 'emsemble':
        genarate_num_dict[modelname] = generate_num
    else:
        genarate_num_dict = {
            'mistral': generate_num,
            'phi': generate_num,
        }

    args = type('Args', (), {
        "prompt_list": prompt_list,
        "conv_mode": None,
        "llava_groups": llava_groups,
        "mids_path": mids_path,
        "eval_samples_num": eval_num,
        "benchmark": benchmark,
        "image_dir": f"benchmark/{benchmark}/imgs",
        "gt": f"benchmark/{benchmark}/{benchmark}.json",
        "save_path": f"results/{benchmark}/ffaa-{modelname}-N{generate_num}-{mids_path.split('/')[2]}-e{epoch}.jsonl",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "generate_num": genarate_num_dict,
        "max_new_tokens": 512
    })()
    eval(args)