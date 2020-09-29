from utils.common import *
from transformers import BertTokenizer, BertModel


def pool_cls(token_embs, token_ids, cls_id, sep_id):
    return token_embs[token_ids == cls_id]


def pool_avg(token_embs, token_ids, cls_id, sep_id):
    cls_b, cls_t = torch.where(token_ids == cls_id)
    sep_b, sep_t = torch.where(token_ids == sep_id)
    assert len(cls_b) == len(sep_b)
    sent_embs = torch.zeros(cls_b.shape[0], token_embs.shape[-1]).to(get_device())
    for si, (cb, ct, sb, st) in enumerate(zip(cls_b, cls_t, sep_b, sep_t)):
        assert cb == sb and ct < st
        sent = token_embs[cb, ct:st + 1, :]
        sent_emb = torch.mean(sent, dim=0)
        sent_embs[si, :] = sent_emb
    return sent_embs


class BertSumExt(nn.Module):
    def __init__(
            self,
            pretrained_bert_model_name='DeepPavlov/rubert-base-cased',
            finetune_bert=False,
            pool='cls',
    ):
        super(BertSumExt, self).__init__()
        self.finetune_bert = finetune_bert

        # do_lower_case=False is crucial
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model_name, do_lower_case=False)
        assert self.tokenizer.pad_token_id == 0
        self.bert = BertModel.from_pretrained(pretrained_bert_model_name)
        self.cls_id = self.tokenizer.cls_token_id

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

        if not self.finetune_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, token_ids, target=None, top_n=None):
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

        if self.finetune_bert:
            token_embs, _ = self.bert(
                input_ids=token_ids,
                attention_mask=(token_ids != self.tokenizer.pad_token_id),
                # token_type_ids=,  # TODO 010101
            )
        else:
            self.bert.eval()  # TODO try without it
            with torch.no_grad():
                token_embs, _ = self.bert(
                    input_ids=token_ids,
                    attention_mask=(token_ids != self.tokenizer.pad_token_id),
                    # token_type_ids=,  # TODO 010101
                )

        sents_embs = self.pool(token_embs, token_ids, self.cls_id, self.tokenizer.sep_token_id)
        sents_logits = self.classifier(sents_embs).squeeze(-1)
        res = (sents_logits,)

        # with open(os.path.join(DATA_PATH, 'sents_embs_3.json'), 'w') as f:
        #     json.dump({
        #         'sents_embs': sents_embs.tolist(),
        #         'target': target.tolist(),
        #     }, f)

        if target is not None:
            assert target.dtype == torch.int64  # TODO consider regression
            loss = nn.BCEWithLogitsLoss()(sents_logits, target.float())
            res += (loss,)

        if top_n is not None:
            num_sentences_by_samples = (token_ids == self.cls_id).sum(dim=1)
            assert sum(num_sentences_by_samples) == len(sents_logits)
            sents_logits_by_samples = -float('inf') * torch.ones(token_ids.shape[0], max(num_sentences_by_samples))
            curr_pos = 0
            for i, num_sentences_in_sample in enumerate(num_sentences_by_samples):
                sents_logits_by_samples[i, :num_sentences_in_sample] = sents_logits[curr_pos:curr_pos + num_sentences_in_sample]
                curr_pos += num_sentences_in_sample

            top_ids = torch.argsort(sents_logits_by_samples, dim=1, descending=True)
            top_ids = top_ids[:, :top_n]
            top_ids = top_ids.sort(dim=1, descending=False).values
            res += (top_ids,)

            if target is not None:
                target_by_samples = torch.zeros_like(sents_logits_by_samples, dtype=torch.int64)
                curr_pos = 0
                for i, num_sentences_in_sample in enumerate(num_sentences_by_samples):
                    target_by_samples[i, :num_sentences_in_sample] = target[curr_pos:curr_pos + num_sentences_in_sample]
                    curr_pos += num_sentences_in_sample

                res += (target_by_samples,)

        return res




















