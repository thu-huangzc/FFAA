from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from llava.model import *
from llava.utils import disable_torch_init
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
from utils.file_utils import *
from utils.llava_utils import *
import random
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_llava(model_path, load_8bit=False, load_4bit=False):
    kwargs = {"device_map": 'auto'}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )
        
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, assign = True)
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    return model, image_processor, tokenizer

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

def eval_model(args):
    # gt jsonl
    gt = read_json(args.gt)
    benchmark = args.benchmark
    # result jsonl
    res_list = []
    if benchmark == 'multiattack':
        count = {'2-class': 0, '4-class': 0}
    else:
        count = 0
    # Model
    disable_torch_init()

    model, image_processor, tokenizer = load_llava(args.model_path, args.load_8bit, args.load_4bit)
    model_name = args.model_name
    prompt_list = args.prompt_list
    conv_mode = args.conv_mode

    processed_num = 0
    for imagename in tqdm(os.listdir(args.image_dir)):
        qs = random.choice(prompt_list)
        prompt = get_llava_prompt(model, qs, conv_mode)
        image = load_image(os.path.join(args.image_dir, imagename))
        image_size = [image.size]
        image_tensor = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_size,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
                pad_token_id=tokenizer.pad_token_id,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        response_json, formatted_response = decode_response(outputs)
    
        result = {}
        result['id'] = imagename
        if benchmark == 'multiattack': # multiattack
            if 'Analysis result' in response_json and 'Forgery type' in response_json:
                result['2-class-correct'] = response_json['Analysis result'] == gt[imagename][0]
                result['4-class-correct'] = response_json['Forgery type'] == gt[imagename][1]
            else:
                result['2-class-correct'] = False
                result['4-class-correct'] = False
            result['content'] = outputs
            if result['2-class-correct']:
                count['2-class'] += 1.0
            if result['4-class-correct']:
                count['4-class'] += 1.0
        else:
            if 'Analysis result' in response_json and 'Forgery type' in response_json:
                result['correct'] = response_json['Analysis result'] == gt[imagename]
            else:
                result['correct'] = False
            result['content'] = outputs
            if result['correct']:
                count += 1.0
        res_list.append(result)
        processed_num += 1

        if processed_num % 10 == 0:
            if benchmark == 'multiattack':
                acc = {'Acc-2-class':count['2-class'] / processed_num, 'Acc-4-class': count['4-class'] / processed_num}
            else:
                acc = {'Acc': count / processed_num}
            res_list.append(acc)
            write_jsonl(args.save_path, res_list)

    if benchmark == 'multiattack':
        acc = {'Acc-2-class':count['2-class'] / processed_num, 'Acc-4-class': count['4-class'] / processed_num}
    else:
        acc = {'Acc': count / processed_num}
    res_list.append(acc)
    write_jsonl(args.save_path, res_list)

if __name__ == "__main__":
    """
        CUDA_VISIBLE_DEVICES=0 python eval_llava.py --benchmark pgc --model_name llava-mistral-7b
    """
    parser = argparse.ArgumentParser(description="llava ffd")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--benchmark', type=str, required=True)
    parser_args = parser.parse_args()

    prompt_list = read_txt_file('playground/prompts.txt')

    benchmark = parser_args.benchmark
    image_dir= f"benchmark/{benchmark}/imgs"
    gt = f"benchmark/{benchmark}/{benchmark}.json"
    model_name = parser_args.model_name

    args = type('Args', (), {
        "model_path": f"checkpoints/{model_name}",
        "model_name": model_name,
        "load_8bit": False,
        "load_4bit": False,
        "prompt_list": prompt_list,
        "conv_mode": 'v1',
        "image_dir": image_dir,
        "gt": gt,
        "benchmark": benchmark,
        "save_path": f"results/{benchmark}/{model_name}_test.jsonl",
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    eval_model(args)