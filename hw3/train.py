#!/usr/bin/env python3

import os.path
import random

import torch
import numpy as np
from transformers import AdamW, GPT2LMHeadModel, get_scheduler

from diallama.mw_loader import Dataset, DataLoader, SPECIAL_TOKENS
from diallama.trainer import Trainer

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

CTX_LEN = 4
BS = 2
EPOCHS = 3
LR = 5e-5
SIZE = 0
MODEL_PATH = os.path.join(os.curdir, 'hw3', 'gpt2-multiwoz')
train_dataset, valid_dataset = Dataset('train', context_len=CTX_LEN, size=SIZE), Dataset('validation', context_len=CTX_LEN, size=SIZE)
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, collate=True)
valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=True, collate=True)

# TODO initialize the model correctly, as well as the optimizer and scheduler
num_training_steps = EPOCHS * len(train_loader)
print("num_training_steps", num_training_steps)

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device type", device)
# model
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
current_num_tokens = model.config.vocab_size
# new_num_tokens = current_num_tokens + len(SPECIAL_TOKENS)
new_num_tokens = current_num_tokens + 2
model.resize_token_embeddings(new_num_tokens)
# optimizer
optimizer = AdamW(model.parameters(), lr=LR)
# scheduler
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

trainer = Trainer(
    model,
    train_loader,
    valid_loader,
    EPOCHS,
    optimizer,
    scheduler
)
trainer.train()
model.save_pretrained(MODEL_PATH)
