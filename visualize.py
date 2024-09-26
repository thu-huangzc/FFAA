import torch
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.file_utils import mask_result
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputTarget
import math
import os

def closest_factors(n):
    sqrt_n = int(math.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return (i, n // i)
    return None

def reshape_transform(tensor):
    # remove [CLS] token
    height, width = closest_factors(tensor.size(1)-1) # should be (24, 24)
    assert height == 24 and width == height

    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # move the channel to the first dimension
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_mids_visual(model_path, texts):
    from mids.mids_visual import MIDSvisual
    model_kwargs = {
        'texts': texts
    }
    model = MIDSvisual(768, **model_kwargs)

    model_state_dict = model.state_dict()

    finetuned_state_dict = torch.load(model_path)
    finetuned_state_dict = {key.replace('module.', ''): value for key, value in finetuned_state_dict.items()}

    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)

    return model.to(dtype=torch.float32)

def get_heatmap(imagepath, answer, cls, layer=1, **kwargs):
    # get args
    checkpoint = kwargs.pop('checkpoint')
    clip_processor = kwargs.pop('clip_processor')
    savename = kwargs.pop('savename')
    dir = kwargs.pop('dir', 'heatmaps')

    masked_answer, _ = mask_result(answer)
    model = load_mids_visual(checkpoint, masked_answer)

    image = Image.open(imagepath).convert('RGB')
    rgb_image = np.float32(image.resize((336,336))) / 255
    input_tensor = clip_processor(images=image, return_tensors="pt")['pixel_values']

    if layer == 1:
        target_layer = model.projector_layer1
    elif layer == 2:
        target_layer = model.projector_layer2
    else:
        NotImplementedError

    # GradCAM visualize heatmap
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(cls)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    heatmap_save_path = os.path.join(dir, "{}_heat{}.png".format(savename, layer))
    cv2.imwrite(heatmap_save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print(f"Heatmap saved to {heatmap_save_path}")
