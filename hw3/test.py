#!/usr/bin/env python3

import os.path
from transformers import GPT2LMHeadModel

# from hw3.train import MODEL_PATH, CTX_LEN, BS
from diallama.mw_loader import Dataset, DataLoader
from diallama.trainer import GenerationWrapper

CTX_LEN = 5
BS = 1
MODEL_PATH = os.path.join(os.curdir, 'hw3', 'gpt2-multiwoz_2_0697')

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
train_dataset, valid_dataset = Dataset('train', context_len=CTX_LEN), Dataset('validation', context_len=CTX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, collate=True)
valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=True, collate=True)
gen_wrapper = GenerationWrapper(model, train_loader.tokenizer, max_length=500)

first100_valid = valid_dataset.data[:100]

# full context
# init_context = ''
# for i, item in enumerate(valid_dataset.data[:100]):
#     if init_context == '' or item['context'][0] == init_context:
#         context = ''
#         for j in range(len(item['context'])):
#             context += " " + item['context'][j]
#         print(i+1, "context>", context)
#         print(i+1, "answer>", gen_wrapper.generate_single(context))
#     else:
#         print(i + 1, "context>", item['context'][0])
#         print(i + 1, "answer>", gen_wrapper.generate_single(item['context'][0]))
#     init_context = item['context'][0]

# first sentence only
init_context = ''
i = 0
for item in valid_dataset.data:
    # if init_context == '':
    #     init_context = item['context'][0]
    #     print(i + 1, "context>", item['context'][0])
    #     print(i + 1, "answer>", gen_wrapper.generate_single(item['context'][0]))
    #     continue
    if item['context'][0] == init_context:
        continue
    else:
        init_context = item['context'][0]
        print(i + 1, "context>", item['context'][0])
        print(i + 1, "answer>", gen_wrapper.generate_single(item['context'][0]))
        i += 1
    if i > 100:
        break

# print(gen_wrapper.generate_single('I am looking for a restaurant in the centre that serves british food.'))
# print(gen_wrapper.generate_single('I want some cheap hotel to stay for 4 nights.'))
# print(gen_wrapper.generate_single('Can you recommend any attractions?'))
