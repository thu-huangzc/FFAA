import re
import json

def read_txt_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def extract_fields(input_string):
    pattern = r'(\w+): ([^\n]+)'
    matches = re.findall(pattern, input_string)
    result = {key: value.strip() for key, value in matches}
    return result

def decode_response(response):
    """
    Get the formatted answer from MLLM's response
    """
    response_json = extract_fields(response)
    if 'description' in response_json:
        response_json['Image description'] = response_json.pop('description')
    if 'reasoning' in response_json:
        response_json['Forgery reasoning'] = response_json.pop('reasoning')
    if 'result' in response_json:
        response_json['Analysis result'] = response_json.pop('result')
    if 'type' in response_json:
        response_json['Forgery type'] = response_json.pop('type')
    key_order = ['Image description', 'Forgery reasoning', 'Analysis result', 'Probability', 'Forgery type']
    sorted_response_json = {key: response_json[key] for key in key_order if key in response_json}
    formatted_response = "\n".join(f"{key}: {value}" for key, value in sorted_response_json.items())
    return sorted_response_json, formatted_response

def mask_result(answer):
    """
    Mask analysis result of the answer
    """
    answer_json, _ = decode_response(answer)
    answer_result = answer_json['Analysis result'].lower()
    new_answer_json = {
        'Image description': answer_json['Image description'],
        'Forgery reasoning': answer_json['Forgery reasoning']
    }
    new_answer = "\n".join(f"{key}: {value}" for key, value in new_answer_json.items())
    return new_answer, answer_result

def answer_format(answer_json):
    """
    whether output formatted answer 
    """
    del answer_json['Probability']
    data = answer_json
    formatted_string = (
        f"Image description: {data['Image description']}\n"
        f"Forgery reasoning: {data['Forgery reasoning']}\n"
        f"Analysis result: {data['Analysis result']}, Forgery type: {data['Forgery type']}\n"
        f"Match score: {data['Match score']}; Difficulty: {data['Difficulty']}"
    )
    return formatted_string

def get_result(answer):
    """
    Get the binary classification result from the anwswer
    """
    answer_json, _ = decode_response(answer)
    answer_result = answer_json['Analysis result'].lower()
    return answer, answer_result

# not much use
def decode_outputs(imgname, gt, outputs):
    result = {}
    response_json, formatted_response = decode_response(outputs)
    result['id'] = imgname
    if 'Analysis result' in response_json and 'Forgery type' in response_json:
        result['2-class-correct'] = response_json['Analysis result'] == gt[imgname][0]
        result['4-class-correct'] = response_json['Forgery type'] == gt[imgname][1]
    else:
        result['2-class-correct'] = False
        result['4-class-correct'] = False
    result['content'] = formatted_response
    return result
    
def read_json(json_file):
    with open(json_file, 'r') as json_f:
        image_info_dict = json.load(json_f)
    return image_info_dict

def load_jsonl(json_file):
    data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def write_jsonl(json_file, data):
    with open(json_file, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

def write_json(json_file, data):
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
