from summarunner.Summarunner import *
from transformers import AdamW
from summarunner.summarunner_data_readers import *


LOWERCASE = True
SENTS_COUNT_MAX = 32
LEN_OF_SENT_MAX = 64
VOCAB_SIZE = 50000
BATCH_SIZE = 16
WORD_EMB = 128
SENT_EMB = 512
DROPOUT = 0.1  # TODO tune!
NUM_EPOCHS = 10
LR = 0.001
# TODO schedule, grad_norm

# SENTS_COUNT_MAX = 3
# LEN_OF_SENT_MAX = 7
# VOCAB_SIZE = 10
# BATCH_SIZE = 4
# WORD_EMB = 2
# SENT_EMB = 2
# LR = 0.001
# DROPOUT = 0.1


def dummy_input():
    inp = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SENTS_COUNT_MAX, LEN_OF_SENT_MAX))
    tgt = torch.zeros(size=(BATCH_SIZE, SENTS_COUNT_MAX), dtype=torch.int64)
    tgt[:, 0] = 1
    return inp, tgt


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    data_iter = tqdm(loader)
    losses, rouges, ious = [], [], []
    for iis, inp, tgt in data_iter:
        out, loss = model(inp.to(get_device()), tgt.to(get_device()))
        losses.append(loss.item())
        data_iter.set_description_str(f'Val loss {loss.item():.03} ')
        probs = torch.sigmoid(out).cpu()
        order = torch.argsort(probs, dim=1, descending=True)
        top = 3
        top_ids_batch = order[:, :top]
        if not (top_ids_batch == torch.tensor([0, 1, 2])).all():
            print(top_ids_batch)

        for ii, top_ids in zip(iis, top_ids_batch):
            top_ids = [int(_) for _ in top_ids]
            text = loader.dataset.items[ii]['text']
            hyp = ' '.join(text[j] for j in sorted(top_ids))
            ref = ' '.join(loader.dataset.items[ii]['summary'])
            rouges.append(calc_rouge(hyp, ref))
            oracle_ids = loader.dataset.items[ii]['oracle']
            ious.append(len(set(oracle_ids).intersection(set(top_ids))) / len(set(oracle_ids).union(set(top_ids))))

    print(f'Mean loss: {np.mean(losses).item():0.03f}')
    print(f'Mean ious: {np.mean(ious).item():0.03f}')
    mean_rouge = calc_mean_rouge(rouges)
    pprint(mean_rouge)


def main(seed=None):
    if seed is not None:
        set_seed(123)

    set_device('cuda')

    train_loader, test_loader, vocabulary = SummarunnerDataset.load_data_gazeta(
        BATCH_SIZE, VOCAB_SIZE, SENTS_COUNT_MAX, LEN_OF_SENT_MAX,
    )

    model = Summarunner(
        vocab_size=VOCAB_SIZE,
        words_emb_size=WORD_EMB,
        sents_emb_size=SENT_EMB,
        padding_idx=vocabulary.PAD_IDX,  # TODO tokenizer
        dropout_p=DROPOUT,
    ).to(get_device())

    model.load_state_dict(torch.load(os.path.join(DATA_PATH, 'rus', 'summarunner.pth')))

    evaluate(model, test_loader)

    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}')
        train_iter = tqdm(train_loader)
        for iis, inp, tgt in train_iter:
            model.train()
            optimizer.zero_grad()
            out, loss = model(inp.to(get_device()), tgt.to(get_device()))
            loss.backward()
            optimizer.step()
            train_iter.set_description_str(f'Train loss {loss.item():.03} ')

        torch.save(model.state_dict(), os.path.join(DATA_PATH, 'rus', f'summarunner_{epoch}.pth'))
        evaluate(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None)
    args = parser.parse_args()
    main(**vars(args))



