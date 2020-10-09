from utils.common import *
from transformers import BertTokenizer, BertModel, BertConfig

from typing import Optional


@torch.jit.script
def pool_cls(token_embs, token_ids, cls_id: int, sep_id: int):
    return token_embs[token_ids == cls_id]


@torch.jit.script
def pool_avg(token_embs, token_ids, cls_id: int, sep_id: int):
    cls_b, cls_t = torch.where(token_ids == cls_id)
    sep_b, sep_t = torch.where(token_ids == sep_id)
    assert len(cls_b) == len(sep_b)
    sent_embs = torch.zeros(cls_b.shape[0], token_embs.shape[-1]).to(token_embs.device)
    for si, (cb, ct, sb, st) in enumerate(zip(cls_b, cls_t, sep_b, sep_t)):
        assert cb == sb and ct < st
        sent = token_embs[cb, ct:st + 1, :]
        sent_emb = torch.mean(sent, dim=0)
        sent_embs[si, :] = sent_emb
    return sent_embs


class BertSumExt(nn.Module):
    @staticmethod
    def create_tokenizer(pretrained_bert_model_name, do_basic_tokenize):
        # do_lower_case=False is crucial
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_bert_model_name,
            do_lower_case=False,
            do_basic_tokenize=do_basic_tokenize
        )
        assert tokenizer.pad_token_id == 0
        return tokenizer

    def __init__(
            self,
            pretrained_bert_model_name,
            do_basic_tokenize,
            finetune_bert=False,
            pool='avg',
            load_pretrained_bert=True,
    ):
        super(BertSumExt, self).__init__()
        self.finetune_bert = finetune_bert
        self.tokenizer = BertSumExt.create_tokenizer(pretrained_bert_model_name, do_basic_tokenize)

        if load_pretrained_bert:
            self.bert = BertModel.from_pretrained(pretrained_bert_model_name)
        else:
            bert_conf = BertConfig.from_pretrained(pretrained_bert_model_name)
            self.bert = BertModel(bert_conf)

        self.cls_id = self.tokenizer.cls_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.sentence_len = self.bert.config.max_position_embeddings

        if pool == 'cls':
            self.pool = pool_cls
        elif pool == 'avg':
            self.pool = pool_avg
        else:
            raise RuntimeError('cls or avg pool only')

        emb_size = self.bert.config.hidden_size

        # self.classifier = nn.Linear(in_features=emb_size, out_features=1)

        # emb_size = emb_size
        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(num_features=emb_size),
        #     nn.Linear(in_features=emb_size, out_features=emb_size * 4),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(num_features=emb_size * 4),
        #     nn.Linear(in_features=emb_size * 4, out_features=1),
        # )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=emb_size),
            nn.Linear(in_features=emb_size, out_features=1),
        )
        torch.nn.init.constant_(self.classifier[1].bias, -1.5)  # initial distribution P(tgt == 1) = 0.18

        self.bcewl_loss = nn.BCEWithLogitsLoss()

        if not self.finetune_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def _prepare_inputs_for_bert(self, input_ids):
        # TODO token_type_ids = 010101..
        return input_ids, input_ids != self.pad_id

    def forward(self, token_ids, target: Optional[torch.Tensor] = None, top_n: Optional[int] = None):
        assert token_ids.shape[1] == self.sentence_len, 'Necessary for tracing: seq_len == max_seq_len'

        # # check model inputs
        # num_sentences_by_samples = (token_ids == self.cls_id).sum(dim=1)
        # curr_pos = 0
        # targets = []
        # for num_sents in num_sentences_by_samples:
        #     targets.append(target[curr_pos:curr_pos + num_sents])
        #     curr_pos += num_sents
        # from bert_sum_ext.bertsumext_eval import print_summarized
        # for tkns, tgt in zip(token_ids, targets):
        #     sentences = self.tokenizer.decode(tkns.cpu()).split('[SEP] [CLS]')
        #     print_summarized(sentences, tgt)
        #     print()

        # TODO try self.bert.eval() if not self.finetune_bert
        token_embs, _ = self.bert(*self._prepare_inputs_for_bert(token_ids))

        sents_embs = self.pool(token_embs, token_ids, self.cls_id, self.sep_id)

        sents_logits = self.classifier(sents_embs).squeeze(-1)
        res = [sents_logits]

        # with open(os.path.join(DATA_PATH, 'sents_embs_3.json'), 'w') as f:
        #     json.dump({
        #         'sents_embs': sents_embs.tolist(),
        #         'target': target.tolist(),
        #     }, f)

        if target is not None:
            assert target.dtype == torch.int64  # TODO consider regression
            loss = self.bcewl_loss(sents_logits, target.float())
            res += [loss]

        if top_n is not None:
            num_sentences_by_samples = (token_ids == self.cls_id).sum(dim=1)
            assert num_sentences_by_samples.sum() == len(sents_logits)
            sents_logits_by_samples = -float('inf') * torch.ones(token_ids.shape[0], max(num_sentences_by_samples))
            curr_pos = 0
            for i, num_sentences_in_sample in enumerate(num_sentences_by_samples):
                sents_logits_by_samples[i, :num_sentences_in_sample] = sents_logits[curr_pos:curr_pos + num_sentences_in_sample]
                curr_pos += num_sentences_in_sample

            top_ids = torch.argsort(sents_logits_by_samples, dim=1, descending=True)
            top_ids = top_ids[:, :top_n]
            top_ids = top_ids.sort(dim=1, descending=False).values
            res += [top_ids]

            if target is not None:
                target_by_samples = torch.zeros_like(sents_logits_by_samples, dtype=torch.int64)
                curr_pos = 0
                for i, num_sentences_in_sample in enumerate(num_sentences_by_samples):
                    target_by_samples[i, :num_sentences_in_sample] = target[curr_pos:curr_pos + num_sentences_in_sample]
                    curr_pos += num_sentences_in_sample

                res += [target_by_samples]

        return res




















