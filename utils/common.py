import random
import re
import string
import torch
import csv
import os
import json
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BartConfig, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from rouge import Rouge
import razdel
from nltk.translate.bleu_score import corpus_bleu
from pprint import pprint
from tqdm import tqdm
from sklearn.model_selection import train_test_split


DEVICE = None
DATA_PATH = 'data/'
MAX_NUM_SAMPLES = None


def set_global_device(device):
    global DEVICE
    DEVICE = device
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
    print('Using', DEVICE)


def get_global_device():
    global DEVICE
    return DEVICE


def set_max_num_samples(max_num_samples):
    global MAX_NUM_SAMPLES
    MAX_NUM_SAMPLES = max_num_samples


def get_max_num_samples():
    global MAX_NUM_SAMPLES
    return MAX_NUM_SAMPLES


def set_seed(seed):
    # note: there are another nuances for gpu and multi-gpu
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_or_create_directory(dir_name):
    """ ignoring all possible errors """
    shutil.rmtree(dir_name, ignore_errors=True)
    cntr = 0
    while True:
        try:
            os.makedirs(dir_name, exist_ok=True)
            return
        except OSError:
            if cntr < 10:
                # some windows bug?
                cntr += 1
                from time import sleep
                sleep(0.1 * cntr)
            else:
                raise


class SummarizationDataset(Dataset):
    def __init__(self, texts, titles):
        self.texts = texts
        self.titles = titles

    def __getitem__(self, item):
        return self.texts[item], self.titles[item]

    def __len__(self):
        return len(self.texts)


def encode_text(tokenizer, texts, max_len=None):
    if isinstance(texts, str):
        texts = [texts]
    assert isinstance(texts, list)
    if max_len is None:
        max_len = 999999999
    enc_texts = [tokenizer.encode(
        txt, return_tensors='pt', max_length=max_len, truncation=max_len is not None).squeeze(0) for txt in texts]
    texts_batch = pad_sequence(enc_texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    return texts_batch


def encode_text_end(tokenizer, texts, max_len=None):
    if isinstance(texts, str):
        texts = [texts]
    assert isinstance(texts, list)
    if max_len is None:
        max_len = 999999999
    enc_texts = []
    for txt in texts:
        enc = tokenizer.encode(txt, return_tensors='pt').squeeze(0)
        enc = torch.cat([torch.tensor([tokenizer.convert_tokens_to_ids('[CLS]')]).long(), enc[-max_len + 1:]])
        enc_texts.append(enc)

    texts_batch = pad_sequence(enc_texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    return texts_batch


class CollateFnStart:
    def __init__(self, tokenizer, max_len_src, max_len_tgt):
        self.tokenizer = tokenizer
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

    def __call__(self, batch):
        return (
            encode_text(self.tokenizer, [txt for txt, title in batch], self.max_len_src),
            encode_text(self.tokenizer, [title for txt, title in batch], self.max_len_tgt)
        )


class CollateFnEnd:
    """ takes end of text """
    def __init__(self, tokenizer, max_len_src, max_len_tgt):
        self.tokenizer = tokenizer
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

    def __call__(self, batch):
        return (
            encode_text_end(self.tokenizer, [txt for txt, title in batch], self.max_len_src),
            encode_text(self.tokenizer, [title for txt, title in batch], self.max_len_tgt)
        )


def decode_text(tokenizer, vocab_ids):
    return tokenizer.decode(
        vocab_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

