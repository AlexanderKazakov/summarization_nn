import random
import re
import string
import torch
from torch import nn
import csv
import os
import argparse
import json
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
# from transformers import BertTokenizer, BartConfig, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from rouge import Rouge
import razdel
# from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
from collections import Counter
from pymystem3 import Mystem
from pprint import pprint
from tqdm import tqdm
from itertools import chain as iter_chain
from contextlib import contextmanager


global_rouge = Rouge()
global_russian_stemmer = RussianStemmer()
global_russian_stopwords = set(stopwords.words("russian"))
global_my_stem = Mystem()


@contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def arg2bool(v):
    if isinstance(v, int):
        if v == 0:
            return False
        elif v == 1:
            return True

    elif isinstance(v, str):
        if v.lower() in ('yes', 'true', '1'):
            return True
        elif v.lower() in ('no', 'false', '0'):
            return False

    raise argparse.ArgumentTypeError('Boolean value expected.')


def calc_rouge(hyp, ref):
    assert isinstance(hyp, str) and isinstance(ref, str)
    assert len(ref.strip()) != 0
    if len(hyp.strip()) != 0:
        return global_rouge.get_scores(hyps=hyp, refs=ref, avg=True)
    else:
        return calc_rouge('x', 'y')  # zeros


def calc_mean_rouge(rouges):
    res = calc_rouge('x', 'y')  # zeros
    for item in rouges:
        for k1, d in res.items():
            for k2 in d:
                res[k1][k2] += item[k1][k2]

    for k1, d in res.items():
        for k2 in d:
            res[k1][k2] /= len(rouges)

    return res


def str_rouge(rg):
    return f"R1 {rg['rouge-1']['f']:.02f}, R2 {rg['rouge-2']['f']:.02f}, RL {rg['rouge-l']['f']:.02f}"


DEVICE = None
DATA_PATH = '../data_summarization/'
CKPT_DIR = 'checkpoints/'
MAX_NUM_SAMPLES = None


def set_device(device):
    global DEVICE
    DEVICE = device
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
    print('Using', DEVICE)


def get_device():
    global DEVICE
    return DEVICE


def set_max_num_samples(max_num_samples):
    global MAX_NUM_SAMPLES
    MAX_NUM_SAMPLES = max_num_samples


def get_max_num_samples():
    global MAX_NUM_SAMPLES
    return MAX_NUM_SAMPLES


def set_data_path(data_path):
    global DATA_PATH
    DATA_PATH = data_path


def get_data_path():
    global DATA_PATH
    return DATA_PATH


def set_ckpt_dir(ckpt_dir):
    global CKPT_DIR
    CKPT_DIR = ckpt_dir


def get_ckpt_dir():
    global CKPT_DIR
    return CKPT_DIR


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


class SimpleVocabulary:
    def __init__(self, all_words, max_vocab_size, pretrained_words=None):
        helper_symbols = ["<PAD>", "<UNK>", "<EOS>"]
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.EOS_IDX = 2

        counts = Counter(all_words)
        print(f'Number of unique input words: {len(counts)}')
        words = [w for w, c in counts.most_common(max_vocab_size)]

        num_words_added = len(helper_symbols)
        if pretrained_words is not None:
            pretrained_words = set(pretrained_words).difference(set(words))
            num_words_added += len(pretrained_words)

        assert max_vocab_size >= num_words_added
        words = words[:-num_words_added]
        print(
            f'SimpleVocabulary:\n'
            f'{len(words)} words from input data,\n'
            f'{len(helper_symbols)} helper words,\n'
            f'{len(pretrained_words) if pretrained_words is not None else 0} pretrained words,'
        )
        words = helper_symbols + words + (pretrained_words if pretrained_words is not None else [])
        print(f'{len(words)} words total')

        self.itos = words
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(tok, self.UNK_IDX) for tok in text] + [self.EOS_IDX]

    def __iter__(self):
        return iter(self.itos)

    def __len__(self):
        return len(self.itos)


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


def nltk_stem_sentence_rus(sentence):
    tokens = word_tokenize(sentence, language='russian')
    tokens = [t for t in tokens if re.search(r'\w', t) is not None and t not in global_russian_stopwords]
    stems = [global_russian_stemmer.stem(t) for t in tokens]
    return ' '.join(stems)


def lemmatize_sentence_rus(sentence):
    lemmas = global_my_stem.lemmatize(sentence)
    lemmas = [t for t in lemmas if re.search(r'\w', t) is not None and t not in global_russian_stopwords]
    return ' '.join(lemmas)


def lemmatize_sentences_rus(sentences):
    """much faster than call lemmatize_sentence_rus in cycle"""
    split = 'fks2hwras1ma39hka766gbk'
    chunk_size = 10000

    def handle_chunk(sentences_chunk):
        all_sents = (' ' + split + ' ').join(sentences_chunk)
        all_lemmas = lemmatize_sentence_rus(all_sents).split()
        chunk_res = [[]]
        for lemma in all_lemmas:
            if lemma == split:
                chunk_res.append([])
            else:
                chunk_res[-1].append(lemma)

        return chunk_res

    res = []
    i = 0
    while i < len(sentences):
        if len(sentences) > chunk_size:
            print(f'Lemmatization: Done for {i} from {len(sentences)} sentences')

        i_step = min(chunk_size, len(sentences) - i)
        res.extend(handle_chunk(sentences[i:i + i_step]))
        i += i_step

    assert len(res) == len(sentences)
    res = [' '.join(arr) for arr in res]
    return res


def lemmatize_texts_rus(texts):
    """split each text to sentences and lemmatize them"""
    sentenized = [[s.text for s in razdel.sentenize(t)] for t in texts]
    texts_lengths = [len(t) for t in sentenized]
    sentences = [s for t in sentenized for s in t]

    sentences_lemm = lemmatize_sentences_rus(sentences)

    texts_lemm = []
    pos = 0
    for text_length in texts_lengths:
        texts_lemm.append(sentences_lemm[pos:pos + text_length])
        pos += text_length

    assert pos == len(sentences)
    assert len(sentenized) == len(texts_lemm)
    assert all(len(s) == len(a) for s, a in zip(sentenized, texts_lemm))
    return texts_lemm, sentenized


def lemmatize_text_rus(text):
    """split text to sentences and lemmatize them"""
    text_lemm, text_sent = lemmatize_texts_rus([text])
    text_lemm, text_sent = text_lemm[0], text_sent[0]
    return text_lemm, text_sent


def get_num_lines_in_file(file_path, *args, **kwargs):
    with open(file_path, *args, **kwargs) as f:
        return sum(1 for _ in f)


class ConsoleColors:
    Map = {
        'PINK': '\033[95m',
        'BLUE': '\033[34m',
        'YELLOW': '\033[93m',
        'RED': '\033[31m',
        'GREEN': '\033[92m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'ITALIC': '\033[3m',
        'ENDCOLOR': '\033[0m',
        '': '\033[0m',
    }

    @staticmethod
    def wrap(string, color):
        return ConsoleColors.Map[color] + string + ConsoleColors.Map['ENDCOLOR']


def print_confusion_matrix(predicted, target, n_classes=None):
    def fmt(val):
        return f'{(val * 100).round(2):>5.1f}'

    def print_sep(sep):
        print(' ' + sep * str_len)

    if n_classes is None:
        n_classes = int(max(max(predicted), max(target)) + 1)

    confusion_matrix = np.zeros((n_classes, n_classes), np.int64)
    assert len(predicted) == len(target)
    for p, t in zip(predicted, target):
        confusion_matrix[t, p] += 1

    confusion_matrix = confusion_matrix / confusion_matrix.sum()

    str_len = 8 * (n_classes + 1) + 11
    row_str = (
        ' | ' +
        ConsoleColors.wrap(fmt(np.diag(confusion_matrix).sum()), 'YELLOW') +
        ' || ' +
        ' | '.join(
            ConsoleColors.wrap(f'p ={i:>2}', 'BLUE') for i in range(n_classes)
        ) +
        ' || ' +
        ConsoleColors.wrap('all p', 'BLUE') +
        ' |'
    )
    print()
    print_sep('-')
    print(row_str)

    for i, row in enumerate(confusion_matrix):
        row_str = (
                ' | ' +
                ConsoleColors.wrap(f't ={i:>2}', 'BLUE') +
                ' || ' +
                ' | '.join(
                    ConsoleColors.wrap(fmt(val), 'YELLOW' if i == j else '') for j, val in enumerate(row)
                ) +
                ' || ' +
                ConsoleColors.wrap(fmt(sum(row)), 'PINK') +
                ' |'
        )
        print_sep('-' if i != 0 else '=')
        print(row_str)

    row_str = (
            ' | ' +
            ConsoleColors.wrap('all t', 'BLUE') +
            ' || ' +
            ' | '.join(
                ConsoleColors.wrap(fmt(val), 'PINK') for val in confusion_matrix.sum(0)
            ) +
            ' || ' +
            ConsoleColors.wrap(fmt(confusion_matrix.sum()), 'YELLOW') +
            ' |'
    )
    print_sep('=')
    print(row_str)
    print_sep('-')
    print()

    return confusion_matrix


def chop_string(user_string, chunk_size=80, join=True):
    output = []
    words = user_string.split(" ")
    total_length = 0

    while total_length < len(user_string) and len(words) > 0:
        line = []
        next_word = words[0]
        line_len = len(next_word) + 1

        while (line_len < chunk_size) and len(words) > 0:
            words.pop(0)
            line.append(next_word)

            if len(words) > 0:
                next_word = words[0]
                line_len += len(next_word) + 1

        line = " ".join(line)
        output.append(line)
        total_length += len(line)

    if join:
        return '\n'.join(output)

    return output







