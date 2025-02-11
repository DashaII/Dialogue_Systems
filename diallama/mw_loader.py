#!/usr/bin/env python3

import pickle
import os

import datasets
import torch.utils.data as torchdata
from torchtext.data.iterator import BucketIterator as TorchBucketIterator
from torch import tensor, int32
import transformers

from diallama.database import MultiWOZDatabase

# DONE: Add special tokens
SPECIAL_TOKENS = ['<|system|>', '<|user|>', '<|endoftext|>', '<|belief|>', '<|database|>']

class Dataset(torchdata.Dataset):
    """
    Dataset class, inherits from torch.utils.data.Dataset.
    Load the MultiWoz dataset using huggingface.datasets
    Able to shorten the context length by setting context_len.
    """
    def __init__(self, split, context_len=None, cache_dir="./", size=0):
        self.split = split
        self.fields = {}
        # Create cache dir if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.f_name = os.path.join(cache_dir, f"{split}_preprocessed_data.json")
        self.database = MultiWOZDatabase()
        # If the dataset has already been preprocessed, load it from the cache
        if os.path.isfile(self.f_name):
            data = pickle.load(open(self.f_name, 'rb'))
            print(f"Loaded {len(data)} examples from cached file.")
        else:
            dataset = datasets.load_dataset(path='multi_woz_v22', split=split, ignore_verifications=True,
                                            streaming=True)
            data = []
            if size == 0:
                for idx, dialogue in enumerate(dataset):
                    if idx % 500 == 0:
                        print(f"Processing dialogue {idx + 1}")
                    data.extend(self.parse_dialogue_into_examples(dialogue, context_len=context_len))
            else:
                for idx, dialogue in enumerate(dataset):
                    if idx < size:
                        if idx % 500 == 0:
                            print(f"Processing dialogue {idx + 1}")
                        data.extend(self.parse_dialogue_into_examples(dialogue, context_len=context_len))
            self.save_data(data)
        self.data = data

    def save_data(self, data):
        assert not os.path.exists(self.f_name), f"{self.f_name} already exists."
        with open(self.f_name, 'wb+') as f:
            pickle.dump(data, f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def parse_dialogue_into_examples(self, dialogue, context_len=None):
        """
        Parses a dialogue into a list of examples.
        Each is a dictionary of the following structure:
        {
            # for HW2:
            'context': list[str],  # list of utterances preceding the current utterance
            'utterance': str,  # the string with the current response
            'delex_utterance': str,  # the string with the current response which is delexicalized, i.e. slot values are
                                    # replaced by corresponding slot names in the text.
            # for HW4:
            'belief_state': dict[str, dict[str, str]],  # belief state dictionary, for each domain a separate belief state dictionary,
                                                        # choose a single slot value if more than one option is available
            'database_results': dict[str, int] # dictionary containing the number of matching results per domain
        }
        The context can be truncated to k last utterances.


        Existing services:
            {'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'}
        Existing intents:
            {'find_bus', 'find_train', 'find_restaurant', 'find_attraction', 'book_hotel', 'find_taxi',
            'find_police', 'book_train', 'find_hotel', 'find_hospital', 'book_restaurant'}
        Existing slots_values_names:
            {'bus-departure', 'hotel-pricerange', 'train-departure', 'hotel-bookstay', 'hotel-bookday',
            'restaurant-bookpeople', 'restaurant-booktime', 'restaurant-pricerange', 'attraction-type',
            'restaurant-name', 'bus-destination', 'train-bookpeople', 'hotel-area', 'taxi-departure',
            'taxi-destination', 'attraction-area', 'attraction-name', 'restaurant-area', 'taxi-arriveby',
            'hotel-stars', 'restaurant-bookday', 'taxi-leaveat', 'hotel-bookpeople', 'restaurant-food',
            'train-destination', 'hospital-department', 'hotel-parking', 'hotel-type', 'train-leaveat',
            'bus-leaveat', 'train-day', 'hotel-name', 'hotel-internet', 'train-arriveby', 'bus-day'}
        """
        examples = []
        turns = dialogue['turns']

        # DONE: create the examples according to the assignment

        ids = turns['turn_id']
        speaker = turns['speaker']
        utterance = turns['utterance']

        for idx in ids:
            idx = int(idx)
            if speaker[idx] == 1:
                dialogue_act = turns['dialogue_acts'][idx]
                slot_names = dialogue_act['span_info']['act_slot_name']
                span_starts = dialogue_act['span_info']['span_start']
                span_ends = dialogue_act['span_info']['span_end']

                delex_utterance = utterance[idx]
                temp_utterance = delex_utterance
                old_end = 0
                for i, (start, end) in enumerate(zip(span_starts, span_ends)):
                    if old_end == 0:
                        temp_utterance = delex_utterance[:start] + "[" + slot_names[i] + "]"
                    else:
                        temp_utterance = temp_utterance + delex_utterance[old_end:start] + "[" + slot_names[i] + "]"
                    old_end = end
                    if i == len(span_ends) - 1:
                        delex_utterance = temp_utterance + delex_utterance[end:]

                # HW4
                frames = turns['frames'][idx-1]
                belief_state = {}
                for i, service in enumerate(frames['service']):
                    belief_state[service] = {}
                    slots_values = frames['state'][i]['slots_values']
                    for (name, value) in zip(slots_values['slots_values_name'], slots_values['slots_values_list']):
                        belief_state[service][name] = value[0]

                database_results = {}
                if belief_state is not None:
                    for domain, constraints in belief_state.items():
                        db_result = self.database.query(domain, constraints)
                        database_results[domain] = len(db_result)

                example = {'context': utterance[:idx], 'utterance': utterance[idx], 'delex_utterance': delex_utterance,
                           'belief_state': belief_state, 'database_results': database_results}
                examples.append(example)

        return examples


class MyBucketIterator(TorchBucketIterator):
    """
    BucketIterator from torchtext.data, overriding the __iter__ method to yield a simple
    batch without PyTorch's Fields.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        """
        Copied from torchtext.data.BucketIterator, but with the following changes:
        `yield minibatch` instead of `yield Batch(minibatch, self.dataset, self.device)`
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield minibatch
            if not self.repeat:
                return


class DataLoader:
    """
    Iteratively returns batches of batch_size from given dataset, optionally shuffled. If collate=True, returns
    integer tokens instead of strings using huggingface.transformers.GPT2Tokenizer.

    Inside a batch, each example has a similar number of tokens, both when tokenizing and when not. To achieve this,
    the sort function is different each time. Slightly edited pytorchtext.legacy.data.BucketIterator is used for bucketing
    the batches and sampling from the batches.
    """

    def __init__(self, dataset, batch_size, shuffle=True, collate=False):
        def _sort_examples_to_buckets_f(example):
            # DONE: implement sorting logic
            total_len = len(example['utterance'].split())
            for context in example['context']:
                total_len += len(context.split())
            return total_len
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate = collate
        # BucketIterator
        self.iterator = MyBucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            sort_key=_sort_examples_to_buckets_f,
            shuffle=shuffle,
            sort_within_batch=True
        )
        self.iterator.create_batches()
        # Tokenizer with special tokens
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS})

    def __iter__(self):
        # Apply collate_fn if desired
        collate = self.collate_fn if self.collate else lambda x: x
        try:
            for batch in self.iterator:
                yield collate(batch)
        except StopIteration:
            self.iterator.create_batches()

    def __len__(self):
        return len(self.iterator)

    def tokenize(self, sentence):
        """
        Uses pretrained GPT2Tokenizer from huggingface.transformers to tokenize a sentence.
        """
        return self.tokenizer(sentence)["input_ids"]

    def collate_fn(self, batch):
        """
        Use transformers.GPT2Tokenizer to convert the batch to a single dictionary (output) of the following structure:

        # for HW2:
        {
        'context': list[list[int]],     # tokenized context (list of subword ids from all preceding dialogue turns,
                                        # system turns prepended with `<|system|>` token and user turns with `<|user|>`)
                                        # for all batch examples
        'utterance': list[list[int]],   # tokenized utterances (list of subword ids from the current dialogue turn)
                                        # for all batch examples
        'delex_utterance': list[list[int]], # tokenized and delexicalized utterances (list of subword ids
                                            # from the current dialogue turn) for all batch examples
        }
        # for HW3, add:
        {
            "input_ids": Tensor[bs, maxlen], # concatenated ids for context and utterance,
                                                # interleaved with the special tokens
            "attention_mask": Tensor[bs, maxlen], # mask, 1 for valid input, 0 for padding
            "context_mask": Tensor[bs, maxlen], # mask, 1 for context tokens, 0 for others
            "utterance_mask": Tensor[bs, maxlen], # mask, 1 for utterance tokens, 0 for others
        }
        # for HW4, add:
        {
        'belief_state': list[list[int]],    # belief state dictionary serialized into a string representation and prepended with
                                            # the `<|belief|>` special token and tokenized (list of subword ids
                                            # from the current dialogue turn) for all batch examples
        'database_results': list[list[int]],    # database result counts serialized into string prepended with the `<|database|>`
                                                # special token and tokenized (list of subword ids from the current dialogue turn)
                                                # for all batch examples
        }
        """
        output = {"context": None,  # hw2
                  "utterance": None,  # hw2
                  "delex_utterance": None,  # hw2
                  "input_ids": None,  # hw3
                  "attention_mask": None,  # hw3
                  "context_mask": None,  # hw3
                  "utterance_mask": None,  # hw3
                  "belief_state": None,  # hw4
                  "database_results": None,  # hw4
                  }

        # HW2
        batch_tokenized_context_list = []
        batch_tokenized_utt_list = []
        batch_tokenized_delex_utt_list = []
        # HW3
        batch_tokenized_input_ids_list = []
        # HW4
        batch_tokenized_belief_state = []
        batch_tokenized_database_results = []
        for example in batch:
            batch_tokenized_utt_list.append(self.tokenizer(example['utterance'])['input_ids'])
            batch_tokenized_delex_utt_list.append(self.tokenizer(example['delex_utterance'])['input_ids'])

            batch_context_str = ""
            for i, context in enumerate(example['context']):
                if i % 2 == 0:
                    context = "<|user|>" + context
                else:
                    context = "<|system|>" + context
                batch_context_str += context
            batch_tokenized_context_list.append(self.tokenizer(batch_context_str)['input_ids'])

            # HW3
            batch_input_ids_str = batch_context_str + "<|endoftext|>" + example['delex_utterance'] + "<|endoftext|>"
            batch_tokenized_input_ids_list.append(self.tokenizer(batch_input_ids_str)['input_ids'])

            # HW4
            belief_state = "<|belief|> { "
            for domain, constr in example['belief_state'].items():
                belief_state += domain + " { "
                for k, v in constr.items():
                    belief_state += k + " : " + v + " , "
                belief_state = belief_state[:-2]
                belief_state += "} "
            belief_state += "}"
            batch_tokenized_belief_state.append(self.tokenizer(belief_state))

            database_results = "<|database|> { "
            for domain, result in example['database_results'].items():
                database_results += domain + " " + str(result) + " , "
            if len(database_results) > 15:
                database_results = database_results[:-2]
            database_results += "}"
            batch_tokenized_database_results.append(self.tokenizer(database_results))

        output['context'] = batch_tokenized_context_list
        output['utterance'] = batch_tokenized_utt_list
        output['delex_utterance'] = batch_tokenized_delex_utt_list

        # HW4
        output['belief_state'] = batch_tokenized_belief_state
        output['database_results'] = batch_tokenized_database_results

        # HW3
        max_len = max(len(sublist) for sublist in batch_tokenized_input_ids_list)
        padded_batch_tokenized_input_ids_list = [sublist + [0]*(max_len - len(sublist)) for sublist in batch_tokenized_input_ids_list]

        context_mask_list = []
        utterance_mask_list = []
        attention_mask_list = []
        for i, input_id in enumerate(padded_batch_tokenized_input_ids_list):
            context_len = len(output['context'][i]) + 1
            context_mask_list.append([1] * context_len + [0] * (max_len - context_len))

            utt_len = len(output['delex_utterance'][i]) + 1
            utterance_mask_list.append([0] * context_len + [1] * utt_len + [0] * (max_len - utt_len - context_len))

            attention_mask_list.append([1] * (context_len + utt_len) + [0] * (max_len - utt_len - context_len))

        output['input_ids'] = tensor(padded_batch_tokenized_input_ids_list, dtype=int32)
        output['context_mask'] = tensor(context_mask_list)
        output['utterance_mask'] = tensor(utterance_mask_list)
        output['attention_mask'] = tensor(attention_mask_list)

        return output

