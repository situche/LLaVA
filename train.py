import re
import json
import torch
import random
import numpy as np
import lightning as L
import torch.nn as nn
from typing import Any, Dict
from nltk import edit_distance
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoProcessor
from lightning.pytorch.callbacks import Callback
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REPO_ID = "nielsr/llava-finetuning-demo"
WANDB_PROJECT = 'LLaVA'
WANDB_NAME = 'llava-demo-cord' 

dataset = load_dataset("naver-clova-ix/cord-v2")

example = dataset['train'][0]
image = example['image']
width, height = image.size
image = image.resize((int(0.3*width), int(0.3*height)))
# print(image)
# print(example['ground_truth'])

ground_truth_json = json.loads(example['ground_truth'])
print(ground_truth_json['gt_parse'])

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = 'right'

USE_LORA = False
USE_QLORA = True

if USE_QLORA or USE_LORA:  # LoRA 或 QLoRA
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_type='nf4', 
            bnb_4bit_compute_dtype=torch.float16
        )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )
else:
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation='flash_attention_2'
    )

def find_all_linear_names(model):
    cls = nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

lora_config = LoraConfig(
    r=8, 
    lora_alpha=8, 
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights='gaussian'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

class LlavaDataset(Dataset):
    def __init__(self, dataset_name_or_path, split='train', sort_json_key=True):
        super().__init__()
        self.split = split
        self.sort_json_key = sort_json_key
        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)
        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample['ground_truth'])
            if 'gt_prases' in ground_truth:
                assert isinstance(ground_truth['gt_prases'], list)
                gt_jsons = ground_truth['gt_parses']
            else:
                assert 'gt_prase' in ground_truth and isinstance(ground_truth['gt_prase'], dict)
                gt_jsons = [ground_truth['gt_prase']]

            self.gt_token_sequences.append(  
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key
                    )
                    for gt_json in gt_jsons
                ]
            )
    
    def json2token(self, obj, sort_json_key=True): 
        if type(obj) == dict:
            if len(obj) == 1 and 'text_sequence' in obj:
                return obj['text_sequence']
            else:
                output = ''
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr'<s_{k}>' + self.json2token(obj[k], sort_json_key) + fr'</s_{k}>'
                    )
                return output
        elif type(obj) == list:
            return r'<sep/>'.join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        sample = self.dataset[index]

        image = sample['image']
        target_sequence = random.choice(self.gt_token_sequences[index])

        return image, target_sequence

train_dataset = LlavaDataset('naver-clova-ix/cord-v2', split='train', sort_json_key=False)
val_dataset = LlavaDataset('naver-clova-ix/cord-v2', split='validation', sort_json_key=False)

train_example = train_dataset[0]
image, target_sequence = train_example
print(target_sequence)

def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        image, ground_truth = example
        images.append(image)

        prompt = f'USER: <image>\nExtract JSON.\nASSISTANT: {ground_truth}'
        texts.append(prompt)
    
    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')

    labels = batch['input_ids'].clone()
    labels['labels' == processor.tokenizer.pad_token_id] = -100
    batch['labels'] = labels

    input_ids = batch['input_ids']
    attention_mask = batch['asstention_mask']
    pixel_values = batch['pixel_values']
    labels = batch['labels']

    return input_ids, attention_mask, pixel_values, labels

def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []
    for example in examples:
        image, ground_truth = example
        images.append(image)

        prompt = f'USER: <image>\nExtract JSON.\nASSISTANT:'
        texts.append(prompt)
        answers.append(ground_truth)
    
    batch = processor(text=texts, images=images, trturn_tensors='pt', padding=True)

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    pixel_values = batch['pixel_values']

    return input_ids， attention_mask， pixel_values

class LlavaModelPLModule(L.LightningModule):
    def __init__(self， config， processor， model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get('batch_size')

    def training_step(self， batch， batch_idx):
        input_ids， attention_mask， pixel_values， labels = batch

        outputs = self.model(input_ids=input_ids， attention_mask=attention_mask， pixel_values=pixel_values， labels=labels)
        loss = outputs.loss

        self.log('train_loss'， loss)
        return loss
    
    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, max_length=MAX_LENGTH)
      
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r'(?:(?=</s))', ''， pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get('verbose', False) 和 len(scores) == 1:
                print(f'Prediction: {pred}')
                print(f'    Answer: {answer}')
                print(f' Normed ED: {scores[0]}')
        
        self.log('val_edit_distance', np.mean(scores))

        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get('lr'))
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

config = {'max_epochs': 10,
          'check_val_every_n_epoch': 1,
          'gradient_clip_val': 1.0,
          'accumulate_grad_batches': 8,
          'lr': 1e-4,
          'batch_size': 2,
          'num_nodes': 1,
          'warmup_steps': 50,
          'result_path': './result',
          'verbose': True}

model_module = LlavaModelPLModule(config, processor, model)

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module):
        print(f'Pushing model to the hub, epoch {trainer.current_epoch}') 
        pl_module.model.push_to_hub(REPO_ID, commir_message=f'Training in progress, epoch {trainer.current_epoch}')
    
    def on_train_end(self, trainer, pl_module):
        print(f'Pushing model to the hub after training')
        pl_module.processor.push_to_hub(REPO_ID, commit_message=f'Training done')
        pl_module.model.push_to_hub(REPO_ID, commit_message=f'Training done')

early_stop_callback = EarlyStopping(monitor='val_edit_distance'， patience=3， varbose=False, mode='min')

wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
    accelerator='gpu',
    devices=[0],
    max_epochs=config.get('max_epochs'),
    accumulate_grad_batches=config.get('accumulate_grad_batches'),
    check_val_every_n_epoch=config.get('check_val_every_n_epoch'),
    gradient_clip_val=config.get('gradient_clip_val'),
    precision='16-mixed',
    limit_val_batches=5,
    num_sanity_val_steps=0,
    logger=wandb_logger,
    callbacks=[PushToHubCallback(), early_stop_callback]
)

trainer.fit(model_module)
