import os
import gc
import copy
import time
import random
import string
import joblib

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint
from torch.cuda.amp import autocast, GradScaler

# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from transformers import DataCollatorWithPadding

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import wandb

from timm.scheduler import CosineLRScheduler

wandb.login()

pd.set_option("max_colwidth", None)
pd.set_option("max_columns", 10)

tqdm.pandas()

torch.backends.cudnn.benchmark = True

def id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

HASH_NAME = id_generator(size=12)
print(HASH_NAME)

BASE_PATH = "/root/kaggle/feedback-prize-effectiveness/data"
TRAIN_DIR = f"{BASE_PATH}/train"
TEST_DIR = f"{BASE_PATH}/test"
MODEL_PATH = f"/root/kaggle/feedback-prize-effectiveness/models/{HASH_NAME}"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    os.makedirs(f"{MODEL_PATH}/tokenizer")

CONFIG = {"seed": 2022,
          "epochs": 3,
          #"model_name": "microsoft/deberta-v3-base",
          "model_name": "microsoft/deberta-v3-large",
          "train_batch_size": 16,
          "valid_batch_size": 16,
          "max_length":    512,
          #"learning_rate": 1e-5, #3e-5, 2e-5
          "learning_rate": 1e-5,
          "warmup_lr_init": 1e-6,
          "scheduler": 'CosineAnnealingWarmRestarts',
          #"scheduler": 'CosineLRScheduler',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 0.1, #1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "num_classes": 3,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          "hash_name": HASH_NAME,
          "competition": "FeedBack",
          "_wandb_kernel": "deb",
          "T_0": 50,
          "n_warmup_epochs": 0.5,
          }

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], use_fast=True)
CONFIG['group'] = f'{HASH_NAME}-Baseline'

additional_special_tokens = {"additional_special_tokens":[
    "[BEFORE_DISCOURSE]", 
    "[AFTER_DISCOURSE]",
    "[CAT_LEAD]",
    "[CAT_POSISION]",
    "[CAT_CLAIM]",
    "[CAT_COUNTERCLAIM]",
    "[CAT_REBUTTAL]",
    "[CAT_EVIDENCE]",
    "[CAT_CONCLUDING_STATEMENT]",
    ]}

tokenizer.add_special_tokens(additional_special_tokens)
print(tokenizer.all_special_tokens)
CONFIG["tokenizer"] = tokenizer
tokenizer.save_pretrained(f"{MODEL_PATH}/tokenizer/")

def get_essay(essay_id):
    essay_path = os.path.join(TRAIN_DIR, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def add_special_tokens(x):
    if x.discourse_type == "Lead":
        return '[CAT_LEAD]'
    elif x.discourse_type == "Position":
        return '[CAT_POSISION]'
    elif x.discourse_type == "Claim":
        return '[CAT_CLAIM]'
    elif x.discourse_type == "Counterclaim":
        return '[CAT_COUNTERCLAIM]'
    elif x.discourse_type == "Rebuttal":
        return '[CAT_REBUTTAL]'
    elif x.discourse_type == "Evidence":
        return '[CAT_EVIDENCE]'
    elif x.discourse_type == "Concluding Statement":
        return '[CAT_CONCLUDING_STATEMENT]'
    else:
        return '[UNK]'

def replace_target_to_special_token(x):
    return x.essay_text.replace(x.discourse_text, '[AFTER_DISCOURSE]')

def short_discourse_text(x):
    tokenizer=CONFIG['tokenizer']
    t = tokenizer(x.discourse_text).input_ids
    if len(t) > 400:
        #print(tokenizer.decode(t[:200] + t[-200:], skip_special_tokens=True))
        return tokenizer.decode(t[:200] + t[-200:], skip_special_tokens=True)
    else:
        return x.discourse_text

def count_short_discourse_text_len(x):
    tokenizer=CONFIG['tokenizer']
    return len(tokenizer(x.short_discourse_text).input_ids)

def short_essay_text(x):
    tokenizer=CONFIG['tokenizer']
    max_len = (508 - x.short_discourse_text_len) // 2
    t = tokenizer(x.essay_text).input_ids
    if len(t) > max_len * 2:
        return tokenizer.decode(t[:max_len] + t[-max_len:], skip_special_tokens=True)
    else:
        return x.essay_text

def count_total_len(x):
    tokenizer=CONFIG['tokenizer']
    return len(tokenizer(x.short_discourse_text).input_ids + tokenizer(x.short_essay_text).input_ids)

df = pd.read_csv(f"{BASE_PATH}/train.csv")
print("=== get essay ===")
df['essay_text'] = df['essay_id'].progress_apply(get_essay)
print("=== add_special_tokens ===")
df['discourse_type_category'] = df.progress_apply(add_special_tokens, axis=1)
#print("=== replace_target_to_special_token ===")
#df['essay_text'] = df.progress_apply(replace_target_to_special_token, axis=1)
print("=== resolve_encodings_and_normalize(discourse_text) ===")
df['discourse_text'] = df['discourse_text'].progress_apply(lambda x : resolve_encodings_and_normalize(x))
print("=== resolve_encodings_and_normalize(essay_text) ===")
df['essay_text'] = df['essay_text'].progress_apply(lambda x : resolve_encodings_and_normalize(x))
print("=== short_discourse_text ===")
df['short_discourse_text'] = df.progress_apply(short_discourse_text, axis=1)
print("=== count_short_discourse_text_len ===")
df['short_discourse_text_len'] = df.progress_apply(count_short_discourse_text_len, axis=1)
print("=== short_essay_text ===")
df['short_essay_text'] = df.progress_apply(short_essay_text, axis=1)
print("=== count_total_len ===")
df['total_len'] = df.progress_apply(count_total_len, axis=1)
print(df.head())
print(df.total_len.describe())

gkf = GroupKFold(n_splits=CONFIG['n_fold'])

for fold, ( _, val_) in enumerate(gkf.split(X=df, groups=df.essay_id)):
    df.loc[val_ , "kfold"] = int(fold)
    
df["kfold"] = df["kfold"].astype(int)
print(df.head())

print(df.groupby('kfold')['discourse_effectiveness'].value_counts())

encoder = LabelEncoder()
df['discourse_effectiveness'] = encoder.fit_transform(df['discourse_effectiveness'])

with open(f"{MODEL_PATH}/le.pkl", "wb") as fp:
    joblib.dump(encoder, fp)

def freeze(module):
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False
        
def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """
    
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters

# 8-bits optimizer
def set_embedding_parameters_bits(embeddings_path, optim_bits=32):
    """
    https://github.com/huggingface/transformers/issues/14819#issuecomment-1003427930
    """
    embedding_types = ("word", "position", "token_type")
    for embedding_type in embedding_types:
        attr_name = f"{embedding_type}_embeddings"
        
        if hasattr(embeddings_path, attr_name): 
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                getattr(embeddings_path, attr_name), 'weight', {'optim_bits': optim_bits}
            )

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse_type_category = df['discourse_type_category'].values
        self.discourse = df['short_discourse_text'].values
        self.essay = df['short_essay_text'].values
        self.targets = df['discourse_effectiveness'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        discourse = self.discourse[index]
        essay = self.essay[index]
        discourse_type_category = self.discourse_type_category[index]
        #text = discourse_type_category + discourse + '[BEFORE_DISCOURSE]' + essay
        text = discourse_type_category + discourse + '[SEP]' + essay
        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len
                    )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[index]
        }

# Dynamic Padding (Collate)
class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

collate_fn = DataCollatorWithPadding(tokenizer=CONFIG['tokenizer'])
#collate_fn = Collate(CONFIG['tokenizer'])

class FeedBackModel(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(FeedBackModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # freezing embeddings and first 6 layers of encoder
        freeze(self.model.embeddings)
        freeze(self.model.encoder.layer[:6])
        freezed_parameters = get_freezed_parameters(self.model)
        print(f"Freezed parameters: {freezed_parameters}")
        self.model.resize_token_embeddings(len(tokenizer))
        (self.model).gradient_checkpointing_enable()
        print(f"Gradient Checkpointing: {(self.model).is_gradient_checkpointing}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, CONFIG['num_classes'])
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.drop(out)
        out = self.drop1(out)
        out = self.drop2(out)
        out = self.drop3(out)
        out = self.drop4(out)
        out = self.drop5(out)
        outputs = self.fc(out)
        return outputs

    def get_parameters(self):
        return filter(lambda parameter: parameter.requires_grad, self.model.parameters())

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    n_steps = len(dataloader)
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(ids, mask)
            loss = criterion(outputs, targets)
            loss = loss / CONFIG['n_accumulate']
        
        scaler.scale(loss).backward()
        #loss.backward()
        
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            #optimizer.step()
            #scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                #scheduler.step((epoch-1) * n_steps + step + 1)
                scheduler.step()

                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)

        outputs = model(ids, mask)
        
        loss = criterion(outputs, targets)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss

def run_training(model, optimizer, scheduler, device, num_epochs, fold):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        
        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{MODEL_PATH}/Loss-Fold-{fold}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def prepare_loaders(fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = FeedBackDataset(df_train, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
    valid_dataset = FeedBackDataset(df_valid, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], collate_fn=collate_fn, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], collate_fn=collate_fn,
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

def fetch_scheduler(optimizer, n_steps):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineLRScheduler':
        scheduler = CosineLRScheduler(optimizer, t_initial=CONFIG['epochs'] * n_steps + 1, lr_min=CONFIG['min_lr'], 
                                  warmup_t=round(n_steps * CONFIG['n_warmup_epochs'] + 1), warmup_lr_init=CONFIG['warmup_lr_init'],
                                  warmup_prefix=True)
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler



for fold in range(0, CONFIG['n_fold']):
    print(f"{y_}====== Fold: {fold} ======{sr_}")
    run = wandb.init(project='FeedBack', 
                     config=CONFIG,
                     job_type='Train',
                     group=CONFIG['group'],
                     tags=[CONFIG['model_name'], f'{HASH_NAME}'],
                     name=f'{HASH_NAME}-fold-{fold}',
                     anonymous='must')
    
    # Create Dataloaders
    train_loader, valid_loader = prepare_loaders(fold=fold)
    
    model = FeedBackModel(CONFIG['model_name'], CONFIG['tokenizer'])
    model.to(CONFIG['device'])
    
    # Define Optimizer and Scheduler
    # selecting parameters, which requires gradients and initializing optimizer
    model_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    #optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    optimizer = AdamW(model.get_parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer, len(train_loader))
    
    model, history = run_training(model, optimizer, scheduler,
                                  device=CONFIG['device'],
                                  num_epochs=CONFIG['epochs'],
                                  fold=fold)
    
    run.finish()
    
    del model, history, train_loader, valid_loader
    _ = gc.collect()
    print()