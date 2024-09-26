import requests
from PIL import Image
from io import BytesIO
import os


def get_image_files_string(directory):
    files = os.listdir(directory)
    image_files = [f"{directory}/{file}" for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files_string = ','.join(image_files)
    return image_files_string

def image_parser(args):
    out = []
    for image in os.listdir(args.image_dir):
        out.append(os.path.join(args.image_dir, image))
    # out = args.image_files.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out