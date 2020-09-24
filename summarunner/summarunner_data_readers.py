from utils.common import *


class SummarunnerDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __getitem__(self, i):
        return i, self.items[i]['text_lemm'], self.items[i]['oracle']

    def __len__(self):
        return len(self.items)

    @staticmethod
    def load_data_gazeta(
            batch_size, max_vocab_size, sents_count_max, len_of_sent_max
    ):
        file_path = os.path.join(DATA_PATH, 'rus', 'gazeta_for_summarunner.jsonl')
        test_ratio = 0.1

        with open(file_path, 'r', encoding='utf-8') as f:
            items = f.readlines()

        items = [json.loads(item) for item in items]

        random.shuffle(items)
        split_i = int(len(items) * (1 - test_ratio))
        assert 0 < split_i < len(items)

        train_items, test_items = items[:split_i], items[split_i:]
        train_words = [w for d in train_items for s in d['text_lemm'] for w in s.split()]
        vocabulary = SimpleVocabulary(train_words, max_vocab_size)

        train_ds = SummarunnerDataset(train_items)
        test_ds = SummarunnerDataset(test_items)
        collator = SummarunnerSimpleVocabularyCollateFn(
            vocabulary, sents_count_max, len_of_sent_max
        )

        num_workers = 0
        return (
            DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=num_workers
            ),
            DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=num_workers
            ),
            vocabulary
        )


class SummarunnerSimpleVocabularyCollateFn:
    def __init__(self, vocabulary, sents_count_max, len_of_sent_max):
        self.vocabulary = vocabulary
        self.sents_count_max = sents_count_max
        self.len_of_sent_max = len_of_sent_max
        assert self.vocabulary.PAD_IDX == 0, "torch.zeros suppose zero padding_idx"

    def __call__(self, inputs):
        batch_size = len(inputs)
        sents_count = max(len(sentences) for ii, sentences, tgt_ids in inputs)
        sents_count = min(sents_count, self.sents_count_max)
        len_of_sent = max(len(sent.split()) for ii, sentences, tgt_ids in inputs for sent in sentences[:sents_count]) + 1  # EOS token
        len_of_sent = min(len_of_sent, self.len_of_sent_max)

        tgt = torch.zeros(batch_size, sents_count, dtype=torch.int64)
        inp = torch.zeros(batch_size, sents_count, len_of_sent, dtype=torch.int64)
        iis = [ii for ii, sentences, tgt_ids in inputs]

        for bi, (ii, sentences, tgt_ids) in enumerate(inputs):
            tgt_ids = [ti for ti in tgt_ids if ti < sents_count]
            tgt[bi, tgt_ids] = 1

            sentences = [s.split()[:len_of_sent - 1] for s in sentences[:sents_count]]
            sentences_enc = [self.vocabulary.encode(s) for s in sentences]
            for si, sent in enumerate(sentences_enc):
                inp[bi, si, :len(sent)] = torch.tensor(sent)

        return iis, inp, tgt



