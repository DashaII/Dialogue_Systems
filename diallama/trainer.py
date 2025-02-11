#!/usr/bin/env python3

from typing import Text, Optional, Dict, Callable, Tuple, List

import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils import ModelOutput
from torch.optim import Optimizer, lr_scheduler
import torch.nn as nn
from logzero import logger

from diallama.mw_loader import DataLoader


class Trainer:
    def __init__(self,
                 model: PreTrainedModel,
                 train_data_loader: DataLoader,
                 valid_data_loader: DataLoader,
                 epochs: int,
                 optimizer: Optimizer,
                 scheduler: lr_scheduler._LRScheduler):
        self.model = model
        self.device = model.device
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        assert epochs > 0
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self):
        logger.info('Starting training...')
        for epoch in range(self.epochs):
            logger.info(f'====== Epoch {epoch}/{self.epochs} Training ======')
            self.model.train()
            for step, batch in enumerate(tqdm.tqdm(self.train_data_loader)):
                """
                The batch is a dictionary of a form:
                {
                    "input_ids": Tensor[bs, maxlen],
                    "attention_mask": Tensor[bs, maxlen],
                    "context_mask": Tensor[bs, maxlen],
                    "utterance_mask": Tensor[bs, maxlen],
                }
                """
                # TODO: Implement model forward and backward steps and all the necessary related logic
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                utterance_mask = batch['utterance_mask'].to(self.device)

                # clone input_ids and set labels for context to -100 (ignored by training)
                # from documentation: All labels set to -100 are ignored (masked), the loss is only computed
                # for labels in [0, ..., config.vocab_size]
                labels = input_ids.clone().long().to(self.device)
                labels[~utterance_mask.bool()] = -100
                # forward step
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # backward step
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if step % 100 == 0:
                    logger.info(f'loss: {output["loss"]}')
            logger.info(f'======= Epoch {epoch}/{self.epochs} Validation ===========')
            valid_loss, valid_token_acc = self.eval()
        logger.info('======= Final Validation ===========')
        final_loss, final_token_acc = self.eval()

    def eval(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        self.model.eval()

        if data_loader is None:
            data_loader = self.valid_data_loader

        total_loss = 0
        total_accuracy = 0
        total_tokens = 0
        for batch in data_loader:
            # TODO: implement evaluation step + Token accuracy & perplexity computation
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            utterance_mask = batch['utterance_mask'].to(self.device)

            labels = input_ids.clone().long().to(self.device)
            labels[~utterance_mask.bool()] = -100

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = output.logits
            total_loss += output.loss.item()

            predictions = torch.argmax(logits, dim=-1)
            # Drop the last prediction
            predictions = predictions[:, :-1]
            # Shift the labels one position to the left
            labels = labels[:, 1:]

            labels_mask = labels != -100
            correct_tokens = (predictions == labels) & labels_mask
            total_accuracy += correct_tokens.sum().item()
            total_tokens += labels_mask.sum().item()

        valid_loss = total_loss/len(data_loader)
        valid_token_acc = total_accuracy/total_tokens
        perplexity = torch.exp(torch.tensor(valid_loss))

        logger.info(f'loss: {valid_loss}')
        logger.info(f'token acc: {valid_token_acc}')
        logger.info(f'perplexity: {perplexity}')

        return valid_loss, valid_token_acc


class GenerationWrapper:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 30):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def interact(self):
        # TODO: Use correct special tokens here, according to what you defined
        while True:
            ctx = []
            user_utterance = input('USER> ')
            user_utterance = user_utterance.strip()
            if user_utterance is None or len(user_utterance) == 0:
                print('Please, provide a nonempty utterance.')
                continue
            if user_utterance.lower() in ['stop', 'end', 'break']:
                break
            input = ' '.join(ctx[-4:]) + '<|endoftext|>' + user_utterance
            response = self.generate_single(user_utterance)
            print(f'SYSTEM> {response}')
            ctx.append('<|user|>' + user_utterance)
            ctx.append('<|system|>' + response)

    def _generate(self, prompts: List[Text]) -> List[Text]:
        self.model.eval()
        # TODO: implement generation from the model and conversion to text
        # prompts to tokens
        input_ids = [self.tokenizer.encode(prompt, return_tensors='pt') for prompt in prompts]
        # generate response
        decoded_outputs = []
        for ids in input_ids:
            outputs = self.model.generate(ids, max_length=self.max_length)
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded_outputs.append(decoded)
        return decoded_outputs

    # force token by token prediction to prevent stopping after eos in the beg of the prediction
    def _generate_step(self, prompts: List[Text]) -> List[Text]:
        self.model.eval()
        # prompts to tokens
        input_ids = [self.tokenizer.encode(prompt, return_tensors='pt') for prompt in prompts]
        # generate response
        decoded_outputs = []
        for ids in input_ids:
            # Initialize the outputs tensor with the input ids
            outputs = ids
            eos = 0
            while True:
                # generate the next token
                next_token_logits = self.model(outputs).logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                # check if the next token is EOS and if yes - stop generation
                if eos > 1:
                    eos = 0
                    break
                if next_token.item() == self.tokenizer.eos_token_id:
                    eos += 1
                # append the next token to the outputs
                outputs = torch.cat([outputs, next_token], dim=-1)

            # decode the outputs to text
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded_outputs.append(decoded)
        return decoded_outputs

    def generate_single(self, prompt: Text) -> Text:
        self.model.config.pad_token_id = self.model.config.eos_token_id

        decoded = self._generate_step([prompt])[0]
        return decoded[len(prompt):]

    def generate_batch(self, prompts: List[Text]) -> List[Text]:
        # see https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
        # OPTIONAL BONUS
        # Implement batch generation
        raise NotImplementedError
