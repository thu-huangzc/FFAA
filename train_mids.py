import torch
from dataclasses import dataclass, field
from PIL import Image
import transformers
from transformers import T5Tokenizer, CLIPProcessor
from torch.utils.data import Dataset
import deepspeed
from typing import Optional
from albumentations import Compose, FancyPCA, HueSaturationValue, OneOf, ImageCompression, GaussNoise, GaussianBlur, MotionBlur
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import datetime
import os
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from utils.mids_utils import flatten_tuple_list
from mids.selector import make_decision_batch

def rank0_print(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)

@dataclass
class ModelArguments:
    hidden_dim: int = 768
    version: str = 'v1'
    image_model_path: str = 'models/clip-vit-large-patch14-336'
    text_model_path: str = 'models/t5-base'


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    val_data_path: str = field(default=None,
                           metadata={"help": "Path to the validation data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    per_device_train_batch_size: int = 24
    per_device_val_batch_size: int = 12
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 1e-5
    num_train_epochs: int = 2
    unfreeze_vision_encoder_last_layers: int = 2
    output_dir: Optional[str] = None

class MLLMAnswersDataset(Dataset):
    def __init__(self, data_path, processor, mode='train'):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.mode = mode
        self.data_dir = os.path.dirname(data_path)
        self.clip_processor = processor
        self.aug = Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.2),
            MotionBlur(p=0.2),
            GaussianBlur(blur_limit=3, p=0.2),
            OneOf([FancyPCA(), HueSaturationValue()], p=0.7),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.mode == 'train':
            image = self.aug(image=image)["image"]
        image = Image.fromarray(image)
        image = self.clip_processor(images=image, return_tensors="pt")['pixel_values']

        cls_label = item['cls_label']
        answers = item['answers']
        texts = [answer['content'] for answer in answers]
        answers_result = [answer['result'] for answer in answers[:3]]
        labels = [answer['label'] for answer in answers]

        return {
            'image': image,
            'cls_label': cls_label,
            'texts': texts,
            'answers_result': answers_result,
            'labels': labels,
        }

def val_model(epoch, model, tokenizer, device, val_dataloader):
    model.eval()

    y_trues = []
    y_preds = []
    y_scores = []

    with tqdm(val_dataloader, desc=f"Val {epoch+1}") as t:
        for batch in t:
            batch_size = len(batch['image'])
            images = batch['image'].squeeze(1).to(device)
            cls_labels = batch['cls_label'].to(device)
            flattened_texts = flatten_tuple_list(batch['texts'])
            answers_result = flatten_tuple_list(batch['answers_result'])
            input_ids = tokenizer(flattened_texts, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True).to(device)

            logits = model(input_ids, images, None, batch_size, 1, 1)['logits']
            scores = F.softmax(logits, dim=2)
            best_answer_idxs, preds, match_scores, forgery_scores = make_decision_batch(answers_result, scores)

            y_trues.extend(cls_labels.detach().cpu().numpy().tolist())
            y_preds.extend(preds)
            y_scores.extend(forgery_scores)
    
    acc = accuracy_score(y_trues, y_preds)
    auc = roc_auc_score(y_trues, y_scores)
    ap = average_precision_score(y_trues, y_scores)

    return acc, auc, ap

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse args
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    deepspeed.init_distributed()
    world_size = torch.distributed.get_world_size()

    # clip processor
    clip_processor = CLIPProcessor.from_pretrained(model_args.image_model_path)
    # dataset
    training_dataset = MLLMAnswersDataset(data_args.data_path, clip_processor, 'train')
    val_dataset = MLLMAnswersDataset(data_args.val_data_path, clip_processor, 'val')
    sampler = DistributedSampler(training_dataset)
    val_sampler = DistributedSampler(val_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=training_args.per_device_train_batch_size, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=training_args.per_device_val_batch_size, sampler=val_sampler)

    # create model
    tokenizer = T5Tokenizer.from_pretrained('models/t5-base', use_fast=False, legacy=False)

    model_kwargs = {
        'loss_type': training_args.loss_type,
        'ratio': training_args.ratio,
        'temperature': training_args.temperature,
        'num_classes': model_args.num_classes,
        'image_model_path': model_args.image_model_path,
        'text_model_path': model_args.text_model_path
    }

    if model_args.version == 'v1':
        from mids.mids_arch import MIDS
        model = MIDS(model_args.hidden_dim, **model_kwargs)  
    else:
        NotImplementedError

    model.freeze_text_encoder_parameters()
    model.freeze_image_encoder_parameters()

    if training_args.unfreeze_vision_encoder_last_layers > 0:
        model.unfreeze_vision_encoder(training_args.unfreeze_vision_encoder_last_layers)
    
    
    def init_non_frozen_params(model):
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear)) or not module.weight.requires_grad:
                continue
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    model.apply(init_non_frozen_params)
    rank0_print("Trainable Parameters:")
    finetuned_layers = ['module.' + name for name, param in model.named_parameters() if param.requires_grad]
    rank0_print(finetuned_layers)
    
    # optim
    optimizer_grouped_parameters = [
        {"params": [p for p in model.parameters() if p.requires_grad]}
    ]   
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    
    # get some common args here
    num_epochs = training_args.num_train_epochs
    output_dir = training_args.output_dir

    # deepspeed config
    ds_config = {
        "train_batch_size": training_args.per_device_train_batch_size * world_size,  # 所有 GPU 上的总批量大小
        "fp16": {
            "enabled": False
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_args.weight_decay,
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_args.learning_rate,
                "warmup_num_steps": int(training_args.warmup_ratio * num_epochs * len(training_dataloader))
            }
        },
        "steps_per_print": 200000
    }

    # init DeepSpeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    # checkpoint save dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f'mids_{model_args.version}_{timestamp}')
    if training_args.unfreeze_vision_encoder_last_layers <= 0:
        save_dir = save_dir+'_freeze'

    rank0_print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_loss = 0

        y_trues = []
        y_preds = []
        y_scores = []

        sampler.set_epoch(epoch)

        # train
        model.train()
        with tqdm(training_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
            for batch in t:
                optimizer.zero_grad()
                batch_size = len(batch['image'])

                images = batch['image'].squeeze(1).to(device)
                cls_labels = batch['cls_label'].to(device)
                flattened_texts = flatten_tuple_list(batch['texts'])
                answers_result = flatten_tuple_list(batch['answers_result'])
                labels = torch.tensor(flatten_tuple_list(batch['labels'])).to(device)
                input_ids = tokenizer(flattened_texts, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True).to(device)
                output = model(input_ids, images, labels, batch_size, 1, 1)

                loss = output['cls_loss']

                model.backward(loss)
                model.step()
                
                epoch_loss += loss.item()

                logits = output['logits']
                scores = F.softmax(logits, dim=2)
                best_answer_idxs, preds, match_scores, forgery_scores = make_decision_batch(answers_result, scores[:,:3,:])

                y_trues.extend(cls_labels.detach().cpu().numpy().tolist())
                y_preds.extend(preds)
                y_scores.extend(forgery_scores)

                train_acc = accuracy_score(y_trues, y_preds)
                train_auc = roc_auc_score(y_trues, y_scores)
                train_ap = average_precision_score(y_trues, y_scores)

                t.set_postfix({'loss': loss.item(), 'acc': train_acc, 'auc': train_auc, 'ap': train_ap})
            
        epoch_loss /= len(training_dataloader)

        val_acc, val_auc, val_ap = val_model(epoch, model, tokenizer, device, val_dataloader)
        rank0_print(f"Val Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - ACC: {val_acc:.4f} - AUC: {val_auc:.4f} - AP: {val_ap:.4f}")
        
        # save the model with the highest val_acc
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            if torch.distributed.get_rank() == 0: 
                finetuned_state_dict = {name: param for name, param in model.state_dict().items() if name in finetuned_layers}
                torch.save(finetuned_state_dict, os.path.join(save_dir, f'{epoch+1}.pth'))
            
    if torch.distributed.get_rank() == 0:
        finetuned_state_dict = {name: param for name, param in model.state_dict().items() if name in finetuned_layers}
        torch.save(finetuned_state_dict, os.path.join(save_dir, f'final.pth'))
    
    rank0_print(f"Best accuracy achieved at epoch {best_epoch+1} with acc {best_acc:.4f}")

if __name__ == "__main__":
    train()