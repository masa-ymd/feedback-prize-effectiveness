import os
import gc
import copy
import time
import random
import string
import joblib
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse

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
import torch.nn.functional as F

# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, AutoModelForTokenClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers.modeling_outputs import ModelOutput, TokenClassifierOutput
from transformers import EarlyStoppingCallback

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

parser = argparse.ArgumentParser(description='train dnn')
parser.add_argument('-f', help='再開するfold', type=int)
parser.add_argument('-t', help='再開する実行時間')
parser.add_argument('-c', help='再開するチェックポイント')
cmdargs = parser.parse_args()

wandb.login()

pd.set_option("max_colwidth", None)
pd.set_option("max_columns", 10)

tqdm.pandas()

torch.backends.cudnn.benchmark = True

jst = timezone(timedelta(hours=+9), 'JST')
tdatetime = datetime.now(jst)
tstr = tdatetime.strftime('%Y-%m-%d_%H:%M:%S')
if cmdargs.t is not None:
    tstr = cmdargs.t

BASE_PATH = "/root/kaggle/feedback-prize-effectiveness/data"
TRAIN_DIR = f"{BASE_PATH}/train"
TEST_DIR = f"{BASE_PATH}/test"
MODEL_PATH = f"/root/kaggle/feedback-prize-effectiveness/models/{tstr}"

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    os.makedirs(f"{MODEL_PATH}/tokenizer")

config = SimpleNamespace()

config.seed = 123
config.model_name = 'microsoft/deberta-v3-large'
config.output_path = Path(MODEL_PATH)
config.input_path = Path('../input/feedback-prize-effectiveness')

config.n_folds = 5
config.lr = 1e-5
config.weight_decay = 0.01
config.epochs = 4
config.batch_size = 36
config.gradient_accumulation_steps = 2
config.warm_up_ratio = 0.1
config.max_len = 512
config.hidden_dropout_prob = 0.1
config.label_smoothing_factor = 0.
config.eval_per_epoch = 3
config.group = f'{tstr}-exp'
config.num_msd = 6

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(config.seed)

tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
tokenizer.model_max_length = config.max_len

additional_special_tokens = {"additional_special_tokens":[
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

def short_discourse_text(x):
    t = tokenizer(x.discourse_text).input_ids
    if len(t) > 400:
        return tokenizer.decode(t[:200] + t[-200:], skip_special_tokens=True)
    else:
        return x.discourse_text

def count_short_discourse_text_len(x):
    return len(tokenizer(x.short_discourse_text).input_ids)

def short_essay_text(x):
    max_len = (508 - x.short_discourse_text_len) // 2
    t = tokenizer(x.essay_text).input_ids
    if len(t) > max_len * 2:
        return tokenizer.decode(t[:max_len] + t[-max_len:], skip_special_tokens=True)
    else:
        return x.essay_text

def count_total_len(x):
    return len(tokenizer(x.short_discourse_text).input_ids + tokenizer(x.short_essay_text).input_ids)

df = pd.read_csv(f"{BASE_PATH}/train.csv")
print("=== get essay ===")
df['essay_text'] = df['essay_id'].progress_apply(get_essay)
print("=== add_special_tokens ===")
df['discourse_type_category'] = df.progress_apply(add_special_tokens, axis=1)
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
#print("=== count_total_len ===")
#df['total_len'] = df.progress_apply(count_total_len, axis=1)
print(df.head())
#print(df.total_len.describe())

cv = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
df['kfold'] = -1
for fold_num, (train_idxs, test_idxs) in enumerate(cv.split(df.index, df.discourse_effectiveness, df.essay_id)):
    df.loc[test_idxs, ['kfold']] = fold_num

#gkf = GroupKFold(n_splits=config.n_folds)
#for fold, ( _, val_) in enumerate(gkf.split(X=df, groups=df.essay_id)):
#    df.loc[val_ , "kfold"] = int(fold)
    
#df["kfold"] = df["kfold"].astype(int)
print(df.groupby('kfold')['discourse_effectiveness'].value_counts())

encoder = LabelEncoder()
df['discourse_effectiveness'] = encoder.fit_transform(df['discourse_effectiveness'])

with open(f"{MODEL_PATH}/le.pkl", "wb") as fp:
    joblib.dump(encoder, fp)

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
        text = discourse_type_category + discourse + '[SEP]' + essay
        inputs = self.tokenizer.encode_plus(
                        text,
                        max_length=self.max_len,
                        truncation=True,
                        padding='max_length',
                        add_special_tokens=True
                    )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': self.targets[index]
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

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
#collate_fn = Collate(CONFIG['tokenizer'])

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

class FeedBackModel(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(FeedBackModel, self).__init__()
        #self.model = AutoModel.from_pretrained(model_name)
        # freezing embeddings and first 6 layers of encoder
        #freeze(self.model.embeddings)
        #freeze(self.model.encoder.layer[:6])
        #freezed_parameters = get_freezed_parameters(self.model)
        #print(f"Freezed parameters: {freezed_parameters}")
        
        self.config = AutoConfig.from_pretrained(model_name, num_labels=3)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config)
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.gradient_checkpointing_enable()
        print(f"Gradient Checkpointing: {(self.model).is_gradient_checkpointing}")

        self.pooler = MeanPooling()
        #self.cnn1 = nn.Conv1d(self.config.hidden_size, 256, kernel_size=2, padding=1)
        #self.cnn2 = nn.Conv1d(256, 3, kernel_size=2, padding=1)
        #self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(self.config.hidden_size, 3)
        #self.fc = nn.Linear(514, 3)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(config.num_msd)])
        
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):        
        out = self.model(input_ids=input_ids,attention_mask=attention_mask,
                         output_hidden_states=output_hidden_states)
        #cnn_out = F.relu(self.cnn1(out.last_hidden_state.permute(0, 2, 1)))
        print(f"out: {out}")
        print(f"type of out: {type(out)}")
        #cnn_out = self.cnn2(cnn_out)
        #lstm_out, _ = self.lstm(out.last_hidden_state, None)
        #sequence_out = lstm_out[:, -1, :]
        #print(f"size {sequence_out.size()}")
        #cat_out = torch.cat([out["hidden_states"][-1*i][:,0] for i in range(1, 4+1)], dim=1)  # concatenate
        #pool_out = self.pooler(out.last_hidden_state, attention_mask)
        pool_out = self.pooler(out[0], attention_mask)
        #logits = sum([self.fc(dropout(sequence_out)) for dropout in self.dropouts]) / config.num_msd
        #print(f"cnn_out2: {self.cnn2(cnn_out).size()}")
        #print(f"{[self.cnn2(dropout(cnn_out)).size() for dropout in self.dropouts]}")
        #logits = sum([self.fc(dropout(pool_out)) for dropout in self.dropouts]) / config.num_msd
        logits = sum([dropout(pool_out) for dropout in self.dropouts]) / config.num_msd
        #logits = logits.view(-1, 3)
        #print(f"logits: {logits.size()}")
        #print(f"labels: {labels.size()}")
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            
        return TokenClassifierOutput(
            logits=logits,
            loss=loss,
            attentions=out['attentions'],
            hidden_states=out['hidden_states']
        )

def criterion(res):
    outputs, labels = res
    loss = nn.CrossEntropyLoss()(
        torch.from_numpy(outputs).to("cuda:0"),
        torch.from_numpy(labels).long().to("cuda:0"))
    return {"loss": loss}

for fold in range(0, config.n_folds):
    
    if cmdargs.f is not None:
        if fold < 2:
            print(f"{y_}====== skip fold {fold} ======{sr_}")
            continue

    print(f"{y_}====== Fold: {fold} ======{sr_}")
    run = wandb.init(project='FeedBack', 
                     config=config,
                     job_type='Train',
                     group=config.group,
                     tags=[config.model_name, f'{tstr}'],
                     name=f'{tstr}-fold-{fold}',
                     anonymous='must')
    
    # Create Dataloaders
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = FeedBackDataset(df_train, tokenizer=tokenizer, max_length=config.max_len)
    valid_dataset = FeedBackDataset(df_valid, tokenizer=tokenizer, max_length=config.max_len)

    num_steps = len(train_dataset) / config.batch_size / config.gradient_accumulation_steps
    eval_steps = num_steps // config.eval_per_epoch
    print(f'Num steps: {num_steps}, eval steps: {eval_steps}')

    args = TrainingArguments(
        output_dir=f"{config.output_path}/fold{fold}",
        learning_rate=config.lr,
        warmup_ratio=config.warm_up_ratio,
        lr_scheduler_type='cosine',
        fp16=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        report_to="wandb",

        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=eval_steps, 
        save_strategy='steps',
        save_steps=eval_steps,

        #metric_for_best_model='loss',
        
        load_best_model_at_end=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        label_smoothing_factor=config.label_smoothing_factor,
        save_total_limit=3  # Prevents running out of disk space.
    )
    
    model = FeedBackModel(config.model_name, tokenizer)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=criterion,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    if cmdargs.c is not None:
        trainer.train(f"{MODEL_PATH}/fold{cmdargs.f}/{cmdargs.c}")
    else:
        trainer.train()
    
    trainer.save_model(f"{config.output_path}/fold{fold}/final")
    
    run.finish()
    
    del model
    _ = gc.collect()
    print()