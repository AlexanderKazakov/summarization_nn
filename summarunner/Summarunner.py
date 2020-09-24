from utils.common import *

# https://arxiv.org/pdf/1611.04230.pdf
# https://github.com/IlyaGusev/summarus/blob/master/summarus/models/summarunner.py


class Summarunner(nn.Module):
    def __init__(
            self,
            vocab_size,
            words_emb_size,
            sents_emb_size,
            padding_idx,
            dropout_p,
    ):
        super(Summarunner, self).__init__()

        self.words_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=words_emb_size,
            padding_idx=padding_idx
        )

        self.words_lstm = nn.LSTM(
            input_size=words_emb_size,
            hidden_size=sents_emb_size,
            num_layers=1,  # TODO 2 with dropout
            bidirectional=False,  # TODO and clarify what is last state for bidirectional lstm and for packed sequence
            batch_first=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(p=dropout_p)

        assert sents_emb_size % 2 == 0
        self.sents_lstm = nn.LSTM(
            input_size=sents_emb_size,
            hidden_size=sents_emb_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )

        self.classifier = nn.Linear(
            in_features=sents_emb_size,
            out_features=1
        )

        # TODO try other magic from article

    def forward(self, token_ids, target=None):
        # B x S x L (S - num sents, L - max sent length, WE/SE - word/sent embedding dimensionality)
        B, S, L = token_ids.shape

        # BS x L
        token_ids = token_ids.reshape(B * S, L)
        # BS x L x WE
        words_emb = self.words_embedding(token_ids)

        # 1 x BS x SE  (1 == num_layers * num_directions)
        _, (sents_emb, _) = self.words_lstm(words_emb)
        # B x S x SE
        sents_emb = sents_emb.reshape(B, S, sents_emb.shape[-1])
        sents_emb = self.dropout(sents_emb)

        # B x S x SE
        sents_emb, _ = self.sents_lstm(sents_emb)
        sents_emb = self.dropout(sents_emb)

        # B x S x 1
        sents_logits = self.classifier(sents_emb)
        # B x S
        sents_logits = sents_logits.squeeze(-1)

        if target is not None:
            assert target.dtype == torch.int64  # TODO consider regression
            loss = nn.BCEWithLogitsLoss()(sents_logits, target.float())
            return sents_logits, loss

        return sents_logits




















