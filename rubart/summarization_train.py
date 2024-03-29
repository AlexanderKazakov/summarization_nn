import sys
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, AdamW, BertTokenizer

sys.path.insert(0, os.getcwd())
from utils.data_utils import *
from rubart.common import *
from rubart.modeling_rubart import BartForConditionalGeneration


def prepare_inputs(source_ids, target_ids):
    encoder_attention_mask = source_ids != tokenizer.pad_token_id
    decoder_input_ids = target_ids[:, :-1].contiguous().clone()
    decoder_attention_mask = decoder_input_ids != tokenizer.pad_token_id
    lm_labels = target_ids[:, 1:].contiguous().clone()
    lm_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100  # nn.CrossEntropyLoss ignore label
    return encoder_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels


def calc_loss(source_ids, target_ids, is_train):
    source_ids, target_ids = source_ids.to(get_device()), target_ids.to(get_device())
    encoder_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels = prepare_inputs(
        source_ids, target_ids
    )
    model.train(is_train)
    output = model(
        input_ids=source_ids,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=lm_labels,
        generation_mode=False
    )
    return output[0]


def train_epoch():
    losses = []
    loader = tqdm(train_loader)
    batch_counter = 0
    total_num_batches = len(train_loader)

    for source_ids, target_ids in loader:
        batch_counter += 1
        if batch_counter > total_num_batches * TRAIN_EPOCH_FRACTION:
            loader.close()
            break

        optimizer.zero_grad()
        loss = calc_loss(source_ids, target_ids, is_train=True)
        loss.backward()
        optimizer.step()
        loss_item = loss.detach().cpu().item()
        loader.set_description(f'{loss_item:.3f}')
        losses.append(loss_item)

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

    return np.mean(losses)


@torch.no_grad()
def val_epoch(loader):
    losses = []
    for source_ids, target_ids in tqdm(loader):
        loss = calc_loss(source_ids, target_ids, is_train=False)
        losses.append(loss.detach().cpu().item())
    return np.mean(losses)


@torch.no_grad()
def test_generation(loader):
    total_rouge = rouge.get_scores(['a'], ['b'], avg=True)  # zero-init
    assert all(v == 0 for m, d in total_rouge.items() for s, v in d.items())
    total_num_samples = 0
    model.eval()
    save_pred, save_tgt, save_inp = [], [], []
    for source_ids, target_ids in tqdm(loader):
        source_ids, target_ids = source_ids.to(get_device()), target_ids.to(get_device())
        encoder_attention_mask, decoder_input_ids, decoder_attention_mask, lm_labels = prepare_inputs(
            source_ids, target_ids
        )
        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=encoder_attention_mask,
            num_beams=4
        )
        predictions = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        targets = [
            tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in target_ids
        ]
        RET_TXT_NUM = 10
        if len(save_pred) < RET_TXT_NUM:
            save_pred.extend(predictions)
            save_tgt.extend(targets)
            save_inp.extend([
                tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for t in source_ids
            ])

        total_num_samples += len(predictions)
        hyp_sents = [[s for s in hyp.split('.') if len(s) > 0] for hyp in predictions]
        tgt_sents = [[s for s in tst.split('.') if len(s) > 0] for tst in targets]
        if any(len(hyp) > 0 and len(tst) > 0 for hyp, tst in zip(hyp_sents, tgt_sents)):  # TODO report bugs to this rouge library
            batch_rouges = rouge.get_scores(predictions, targets, avg=False, ignore_empty=True)
            for m in total_rouge:
                for d in total_rouge[m]:
                    for sample_rouge in batch_rouges:
                        prev_val = total_rouge[m][d]
                        total_rouge[m][d] += sample_rouge[m][d]
                        assert prev_val < total_rouge[m][d] or sample_rouge[m][d] == 0  # check overflow

    for m in total_rouge:
        for d in total_rouge[m]:
            total_rouge[m][d] /= total_num_samples

    return total_rouge, save_pred, save_tgt, save_inp


def fit():
    last_ckpt_dir = os.path.join(get_ckpt_dir(), 'last_ckpt/')
    best_ckpt_dir = os.path.join(get_ckpt_dir(), 'best_ckpt/')
    clear_or_create_directory(get_ckpt_dir())
    model.to(get_device())
    all_epoch_statistics = []
    min_val_loss = np.inf
    for epoch in range(1000):
        print(f'Epoch {epoch}')
        train_loss = train_epoch()
        print(f'Train loss: {train_loss}')
        clear_or_create_directory(last_ckpt_dir)
        model.save_pretrained(last_ckpt_dir)
        statistics = {'epoch': epoch, 'train_loss': train_loss}
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(train_loss)

        if epoch % 2 == 0:
            val_loss = val_epoch(val_loader)
            print(f'Validation loss: {val_loss}')
            statistics['val_loss'] = val_loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                clear_or_create_directory(best_ckpt_dir)
                model.save_pretrained(best_ckpt_dir)

        if epoch % 6 == 0 and epoch != 0:
            total_rouge, predictions, targets, inputs = test_generation(val_loader)
            statistics['total_rouge'], statistics['predictions'], statistics['targets'], statistics['inputs'] = \
                total_rouge, predictions, targets, inputs
            pprint(total_rouge)
            # for hyp, ref, inp in zip(predictions, targets, inputs):
            #     print(ref)
            #     print(hyp)
            #     print('-' * 50)
            #     print(inp)
            #     print('=' * 50 + '\n')

        all_epoch_statistics.append(statistics)
        with open(os.path.join(get_ckpt_dir(), 'statistics.json'), 'w') as f:
            json.dump(all_epoch_statistics, f, indent=4, sort_keys=True)

    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=24,
        type=int,
    )
    parser.add_argument(
        "--min_target_length",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--max_num_samples",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        default='cuda',
        type=str,
        help="cuda or cpu",
    )
    parser.add_argument(
        "--init_ckpt_dir",
        default=None,
        type=str,
        help="checkpoint folder to load weights from. "
             "default -- just load weights from rubert to encoder, initialize decoder with norm(0.02) ",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--train_whole_model",
        default=False,
        type=bool,
        help="train all parameters. if False (default), train only decoder layers"
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        type=str,
        help="type of scheduler: None, plateau_decay or linear_decay",
    )
    parser.add_argument(
        "--scheduler_num_epochs",
        default=None,
        type=int,
        help="patience for plateau_decay; epochs to zero for linear_decay",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="gazeta, lenta, wikihow, etc"
    )
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="base path to datasets",
    )
    parser.add_argument(
        "--ckpt_dir",
        default=None,
        type=str,
        help="path to save checkpoints during the training",
    )

    TRAIN_EPOCH_FRACTION = 0.2
    args = parser.parse_args()
    set_device(args.device)
    set_seed(args.seed)
    set_max_len_src(args.max_source_length)
    set_max_len_tgt(args.max_target_length)
    set_min_len_tgt(args.min_target_length)
    set_max_num_samples(args.max_num_samples)
    if args.data_path is not None:
        set_data_path(args.data_path)
    if args.ckpt_dir is not None:
        set_ckpt_dir(args.ckpt_dir)

    print(args)

    if args.init_ckpt_dir is not None:
        assert os.path.isdir(args.init_ckpt_dir)
        model = BartForConditionalGeneration.from_pretrained(args.init_ckpt_dir)
        model.config.min_length = get_min_len_tgt()
        model.config.max_length = get_max_len_tgt()
        tokenizer = BertTokenizer.from_pretrained(
            args.init_ckpt_dir, do_lower_case=False)  # do_lower_case=False is crucial
    else:
        model, tokenizer = load_rubart_with_pretrained_encoder()

    # if args.dataset == 'sportsru':
    #     train_loader, val_loader, test_loader = read_sportsru(CollateFnEnd(tokenizer))
    # else:
    train_loader, val_loader = read_dataset_to_loaders(
        args.dataset, args.batch_size,
        CollateFnStart(tokenizer, get_max_len_src(), get_max_len_tgt()), get_max_len_src(), get_max_len_tgt())

    params = model.parameters() if args.train_whole_model else model.model.decoder.layers.parameters()
    optimizer = AdamW(params, lr=args.lr)
    # TODO weight decay with param groups
    if args.scheduler is not None:
        assert args.scheduler_num_epochs is not None
        if args.scheduler == 'plateau_decay':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_num_epochs, verbose=True)
        else:
            assert args.scheduler == 'linear_decay'
            num_steps = int(TRAIN_EPOCH_FRACTION * args.scheduler_num_epochs * len(train_loader)) + 10
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_steps)

    rouge = Rouge()
    fit()



