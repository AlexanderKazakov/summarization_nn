from utils.common import *


class BertSumExtDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __getitem__(self, i):
        return i, self.items[i]['text'], self.items[i]['oracle']

    def __len__(self):
        return len(self.items)

    @staticmethod
    def load_data_gazeta(
            data_path, batch_size, tokenizer, max_text_len, num_workers, single_batch, data_split_seed,
    ):
        file_path = os.path.join(data_path, 'gazeta_for_extractive.jsonl')
        test_ratio = 0.1

        with open(file_path, 'r', encoding='utf-8') as f:
            items = f.readlines()

        if single_batch:
            items = items[:batch_size]
            items = [json.loads(item) for item in items]
            train_ds = BertSumExtDataset(items)
            test_ds = BertSumExtDataset(items)

        else:
            items = [json.loads(item) for item in items]

            with temp_np_seed(data_split_seed):
                np.random.shuffle(items)
                assert (np.random.randint(10, size=10) == [5, 9, 4, 2, 4, 6, 8, 1, 0, 3]).all(), 'set data seed to 123'

            split_i = int(len(items) * (1 - test_ratio))
            assert 0 < split_i < len(items)

            train_items, test_items = items[:split_i], items[split_i:]

            train_ds = BertSumExtDataset(train_items)
            test_ds = BertSumExtDataset(test_items)

        collator = BertSumExtCollateFn(
            tokenizer,
            max_text_len,
        )
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
            )
        )


class BertSumExtCollateFn:
    def __init__(self, tokenizer, max_text_len):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def get_num_encoded_sentences(self, sentences):
        """for greedy oracle. same algo as in __call__"""
        curr_pos = 0
        for sent_i, sent in enumerate(sentences):
            max_sent_len = self.max_text_len - curr_pos
            sent_enc = self.tokenizer.encode(sent, max_length=max_sent_len,
                                             truncation=True, padding=False, return_tensors='pt').squeeze(0)
            curr_pos += len(sent_enc)
            if curr_pos >= self.max_text_len - 2:  # -2 for CLS and SEP
                break

        return sent_i + 1

    def __call__(self, inputs):
        batch_size = len(inputs)
        inp = torch.zeros(batch_size, self.max_text_len, dtype=torch.int64)
        tgt = []
        sents_count = 0
        for bi, (ii, sentences, tgt_ids) in enumerate(inputs):
            curr_pos = 0
            for sent_i, sent in enumerate(sentences):
                max_sent_len = self.max_text_len - curr_pos
                sent_enc = self.tokenizer.encode(sent, max_length=max_sent_len,
                                                 truncation=True, padding=False, return_tensors='pt').squeeze(0)
                assert sent_enc[0] == 101 and sent_enc[-1] == 102
                inp[bi, curr_pos:curr_pos + len(sent_enc)] = sent_enc
                tgt.append(int(sent_i in tgt_ids))
                curr_pos += len(sent_enc)
                if curr_pos >= self.max_text_len - 2:  # -2 for CLS and SEP
                    break

            sents_count = max(sents_count, sent_i + 1)

        iis = [ii for ii, sentences, tgt_ids in inputs]
        tgt = torch.tensor(tgt)
        return iis, inp, tgt



