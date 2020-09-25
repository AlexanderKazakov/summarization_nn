from utils.common import *
from transformers import BertTokenizer, BertModel


class BertSumExt(nn.Module):
    def __init__(
            self,
            finetune_bert,
    ):
        super(BertSumExt, self).__init__()
        self.finetune_bert = finetune_bert

        rubert_ckpt_name = 'DeepPavlov/rubert-base-cased'
        self.tokenizer = BertTokenizer.from_pretrained(rubert_ckpt_name, do_lower_case=False)  # do_lower_case=False is crucial
        assert self.tokenizer.pad_token_id == 0
        self.bert = BertModel.from_pretrained(rubert_ckpt_name)
        self.cls_id = self.tokenizer.cls_token_id

        self.classifier = nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)

        if not self.finetune_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, token_ids, target=None):
        if self.finetune_bert:
            self.bert.train()
            token_embs, _ = self.bert(
                input_ids=token_ids,
                attention_mask=(token_ids != self.tokenizer.pad_token_id),
                # token_type_ids=,  # TODO 010101
            )
        else:
            self.bert.eval()
            with torch.no_grad():
                token_embs, _ = self.bert(
                    input_ids=token_ids,
                    attention_mask=(token_ids != self.tokenizer.pad_token_id),
                    # token_type_ids=,  # TODO 010101
                )

        sents_embs = token_embs[token_ids == self.cls_id]
        sents_logits = self.classifier(sents_embs).squeeze(-1)

        if target is not None:
            assert target.dtype == torch.int64  # TODO consider regression
            loss = nn.BCEWithLogitsLoss()(sents_logits, target.float())
            return sents_logits, loss

        return sents_logits




















