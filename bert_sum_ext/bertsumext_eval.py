from bert_sum_ext.BertSumExt import *
from bert_sum_ext.bertsumext_data_readers import *


class LeadN(nn.Module):
    def __init__(self):
        super(LeadN, self).__init__()

    def forward(self, token_ids=None, target=None, top_n=None):
        assert top_n is not None
        batch_size = token_ids.shape[0]
        top_ids = torch.arange(top_n).repeat(batch_size, 1)
        return None, torch.zeros(1), top_ids, None


class Oracle(nn.Module):
    def __init__(self, cls_id):
        super(Oracle, self).__init__()
        self.cls_id = cls_id

    def forward(self, token_ids=None, target=None, top_n=None):
        batch_size = token_ids.shape[0]
        num_sentences_by_samples = (token_ids == self.cls_id).sum(dim=1)
        target_by_samples = torch.zeros((batch_size, max(num_sentences_by_samples)), dtype=torch.int64)
        curr_pos = 0
        for i, num_sentences_in_sample in enumerate(num_sentences_by_samples):
            target_by_samples[i, :num_sentences_in_sample] = target[curr_pos:curr_pos + num_sentences_in_sample]
            curr_pos += num_sentences_in_sample

        top_ids = [torch.where(_ > 0)[0] for _ in target_by_samples]
        return None, torch.zeros(1), top_ids, target_by_samples


@torch.no_grad()
def evaluate(model, loader, top_n, verbose, lowercase_rouge):
    model.eval()
    data_iter = tqdm(loader) if verbose else loader
    losses, rouges, ious = [], [], []
    target_hist, model_hist = np.zeros(256), np.zeros(256)
    for iis, inp, tgt in data_iter:
        out, loss, top_ids_by_samples, target_by_samples = model(inp.to(get_device()), tgt.to(get_device()), top_n=top_n)
        losses.append(loss.item())
        if verbose:
            data_iter.set_description_str(f'Val loss {loss.item():.03} ')

        dataset_items = [loader.dataset.items[i] for i in iis]
        if target_by_samples is None:
            target_by_samples = [None for _ in dataset_items]

        for sample_target, sample_top_ids, sample in zip(target_by_samples, top_ids_by_samples, dataset_items):
            init_oracle_ids = sample['oracle']
            target_ids = torch.where(sample_target > 0)[0] if sample_target is not None else torch.tensor(init_oracle_ids)
            # assert (torch.tensor(init_oracle_ids[:len(target_ids)]) == target_ids).all()
            assert (torch.tensor(init_oracle_ids) == target_ids).all()
            for i in target_ids:
                target_hist[i] += 1
            for i in sample_top_ids:
                model_hist[i] += 1

            hyp = ' '.join(sample['text'][_] for _ in sample_top_ids)
            ref = ' '.join(sample['summary'])
            if lowercase_rouge:
                hyp = hyp.lower()
                ref = ref.lower()

            rg = calc_rouge(hyp, ref)
            rouges.append(rg)

            # tgt = ' '.join(sample['text'][_] for _ in target_ids)
            # rg_tgt = calc_rouge(tgt, ref)
            # print('=' * 20)
            # pprint('ref: ' + ref)
            # print(f'init oracle: {init_oracle_ids}, tgt: {target_ids}, hyp: {sample_top_ids}')
            # print('-' * 20)
            # pprint(f'hyp ({str_rouge(rg)}): ' + hyp)
            # print('-' * 20)
            # pprint(f'tgt ({str_rouge(rg_tgt)}): ' + tgt)
            # print('-' * 20)
            # print()

            top_ids = [int(_) for _ in sample_top_ids]
            target_ids = [int(_) for _ in target_ids]
            ious.append(len(set(target_ids).intersection(set(top_ids))) / len(set(target_ids).union(set(top_ids))))

    mean_loss = np.mean(losses).item()
    mean_iou = np.mean(ious).item()
    mean_rouge = calc_mean_rouge(rouges)
    return (
        mean_loss,
        mean_iou,
        mean_rouge,
        model_hist,
        target_hist,
    )


def print_summarized(sentences, probs):
    sent_order = torch.argsort(probs, descending=True).tolist()
    print(f'Total {len(sentences)} sentences. Probabilities:')
    print([f'{v:.03f}' for v in probs[sent_order]])

    for sent_i, sent in enumerate(sentences):
        if sent_i == len(probs):
            print('-' * 80)

        if sent_i >= len(probs):
            color = ''
        elif len(sent_order) > 0 and sent_i == sent_order[0]:
            color = 'YELLOW'
        elif len(sent_order) > 1 and sent_i == sent_order[1]:
            color = 'PINK'
        elif len(sent_order) > 2 and sent_i == sent_order[2]:
            color = 'RED'
        elif probs[sent_i] >= probs[len(probs) // 2]:
            color = 'BLUE'
        else:
            color = ''

        print(ConsoleColors.wrap(chop_string(
            f'({(probs[sent_i] if sent_i < len(probs) else 0) * 100:2.0f}) {sent}'), color))

    print()
    print('=' * 35 + ' 3-sentence summary: ' + '=' * 35)
    res = ' '.join(sentences[i] for i in sorted(sent_order[:3]))
    print(chop_string(res))
    print()


@torch.no_grad()
def infer(tokenizer, model, text):
    model.eval()
    collator = BertSumExtCollateFn(
        tokenizer,
        512  # for tracing, model.sentence_len is not working TODO
    )
    sentences = sentenize_with_newlines(text)
    _, inp, _ = collator([(0, sentences, [])])
    # sent_logits, top_ids = model(inp.to(get_device()), None, top_n=len(sentences) // 2)
    # top_ids = top_ids.squeeze(0).tolist()
    sent_logits, = model(inp.to(get_device()))

    probs = torch.sigmoid(sent_logits.cpu())
    # sent_order = torch.argsort(probs, descending=True).tolist()
    # assert sorted(sent_order[:len(sentences) // 2]) == top_ids
    print_summarized(sentences, probs)


if __name__ == '__main__':
    set_device('cpu')
    set_seed(123)

    data_path = 'data'
    batch_size = 8
    use_traced = True
    model_file = 'bertsumext_40000_30_09'
    pretrained_bert_model_name = 'DeepPavlov/rubert-base-cased-sentence'
    ckpt_path = os.path.join(data_path, 'rus', 'gazeta', model_file + '.{}')

    if use_traced:
        model = torch.jit.load(ckpt_path.format('torchscript'), map_location=get_device())
        model.eval()
        tokenizer = BertSumExt.create_tokenizer(pretrained_bert_model_name)
    else:
        model = BertSumExt(
            pretrained_bert_model_name=pretrained_bert_model_name,
            finetune_bert=False,
            pool='avg',  # TODO configs!!!
        )
        model.to('cpu')  # to avoid cuda out of memory while loading
        model.load_state_dict(torch.load(ckpt_path.format('pth')))
        model.to(get_device())
        tokenizer = model.tokenizer

    # with open(ckpt_path.format('json'), encoding='utf-8') as f:
    #     statistics = json.load(f)
    #     plot_sents_hist(statistics['model_hist'], statistics['target_hist'])

    infer_dir = os.path.join(data_path, 'rus/my_inputs')
    fnames = next(os.walk(infer_dir))[2]
    # fnames = ['3.txt']
    for fname in fnames:
        print(fname)
        with open(os.path.join(infer_dir, fname), 'r', encoding='utf-8') as f:
            text = f.read()
        infer(tokenizer, model, text)
        print('\n\n')

    # train_loader, test_loader = BertSumExtDataset.load_data_gazeta(
    #     data_path, batch_size, model.tokenizer, model.bert.config.max_position_embeddings, 0, 0, 123,
    # )
    #
    # mean_loss, mean_iou, mean_rouge, model_hist, target_hist = evaluate(model, test_loader, 3, True, False)
    #
    # # lead_n = LeadN()
    # # mean_loss, mean_iou, mean_rouge, model_hist, target_hist = evaluate(lead_n, test_loader, 3, True, False)
    #
    # # oracle = Oracle(model.cls_id)
    # # mean_loss, mean_iou, mean_rouge, model_hist, target_hist = evaluate(oracle, test_loader, 3, True, False)
    #
    # print(f'Mean loss: {mean_loss:0.03f}')
    # print(f'Mean ious: {mean_iou:0.03f}')
    # pprint(mean_rouge)
    # plot_sents_hist(model_hist, target_hist)










