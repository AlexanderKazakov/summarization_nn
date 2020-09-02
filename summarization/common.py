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


def set_global_device(device):
    global DEVICE
    DEVICE = device
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
    print('Using', DEVICE)

def get_global_device():
    global DEVICE
    return DEVICE


def set_batch_size(batch_size):
    global BATCH_SIZE
    BATCH_SIZE = batch_size

def get_batch_size():
    global BATCH_SIZE
    return BATCH_SIZE

def set_max_len_src(max_len_src):
    global MAX_LEN_SRC
    MAX_LEN_SRC = max_len_src

def get_max_len_src():
    global MAX_LEN_SRC
    return MAX_LEN_SRC

def set_max_len_tgt(max_len_tgt):
    global MAX_LEN_TGT
    MAX_LEN_TGT = max_len_tgt

def get_max_len_tgt():
    global MAX_LEN_TGT
    return MAX_LEN_TGT

def set_min_len_tgt(min_len_tgt):
    global MIN_LEN_TGT
    MIN_LEN_TGT = min_len_tgt

def get_min_len_tgt():
    global MIN_LEN_TGT
    return MIN_LEN_TGT


def set_small_run(small_run):
    global SMALL_RUN
    SMALL_RUN = small_run

def get_small_run():
    global SMALL_RUN
    return SMALL_RUN


def set_seed(seed):
    # note: there are another nuances for gpu and multi-gpu
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# logging.basicConfig(level=logging.INFO)

DATA_PATH = 'data/'
CKPT_DIR = 'rubart_checkpoints/'
RUBART_ENC_WEIGHTS_DIR = DATA_PATH + 'ckpts/rubart_initial_weights_from_rubert/'

BATCH_SIZE = None

# for lenta data optimal is 512 / 24, for ria data -- 1024 / 24, for sportsru -- 4000 / 200
MAX_LEN_SRC = None
MAX_LEN_TGT = None
MIN_LEN_TGT = None

SMALL_RUN = False


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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len_src = get_max_len_src()
        self.max_len_tgt = get_max_len_tgt()

    def __call__(self, batch):
        return (
            encode_text(self.tokenizer, [txt for txt, title in batch], self.max_len_src),
            encode_text(self.tokenizer, [title for txt, title in batch], self.max_len_tgt)
        )


class CollateFnEnd:
    """ takes end of text """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len_src = get_max_len_src()
        self.max_len_tgt = get_max_len_tgt()

    def __call__(self, batch):
        return (
            encode_text_end(self.tokenizer, [txt for txt, title in batch], self.max_len_src),
            encode_text(self.tokenizer, [title for txt, title in batch], self.max_len_tgt)
        )


def decode_text(tokenizer, vocab_ids):
    return tokenizer.decode(
        vocab_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)


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


def load_rubart_with_pretrained_encoder():
    from summarization.modeling_rubart import BartForConditionalGeneration

    tokenizer = BertTokenizer.from_pretrained(RUBART_ENC_WEIGHTS_DIR, do_lower_case=False)  # do_lower_case=False is crucial
    config = BartConfig.from_pretrained(RUBART_ENC_WEIGHTS_DIR)
    config.task_specific_params = None
    config.min_length, config.max_length = get_min_len_tgt(), get_max_len_tgt()
    print(config)

    model = BartForConditionalGeneration(config)
    model.model.encoder.load_state_dict(torch.load(RUBART_ENC_WEIGHTS_DIR + 'encoder_state_dict.pth'))
    # embeddings sharing
    model.model.decoder.embed_positions.weight = model.model.encoder.embed_positions.weight
    model.model.decoder.token_type_embeddings.weight = model.model.encoder.token_type_embeddings.weight
    model.model.decoder.layernorm_embedding.weight = model.model.encoder.layernorm_embedding.weight
    model.model.decoder.layernorm_embedding.bias = model.model.encoder.layernorm_embedding.bias
    assert (model.model.shared.weight == model.model.encoder.embed_tokens.weight).all()
    assert (model.model.shared.weight == model.model.decoder.embed_tokens.weight).all()
    assert (model.model.encoder.embed_positions.weight == model.model.decoder.embed_positions.weight).all()
    assert (model.model.encoder.token_type_embeddings.weight == model.model.decoder.token_type_embeddings.weight).all()
    assert (model.model.encoder.layernorm_embedding.weight == model.model.decoder.layernorm_embedding.weight).all()
    assert (model.model.encoder.layernorm_embedding.bias == model.model.decoder.layernorm_embedding.bias).all()

    # the only not pretrained parameters are decoder.layers
    return model, tokenizer







