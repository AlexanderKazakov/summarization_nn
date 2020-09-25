from bert_sum_ext.BertSumExt import *
from bert_sum_ext.bertsumext_data_readers import *
from transformers import AdamW, get_linear_schedule_with_warmup


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    data_iter = tqdm(loader)
    losses, rouges, ious = [], [], []
    for iis, inp, tgt in data_iter:
        out, loss = model(inp.to(get_device()), tgt.to(get_device()))
        losses.append(loss.item())
        data_iter.set_description_str(f'Val loss {loss.item():.03} ')
        num_sentences = (inp == model.cls_id).sum(dim=1)
        assert sum(num_sentences) == len(out)
        probs = torch.sigmoid(out).cpu()

        curr_pos = 0
        for txt_i, txt_len in enumerate(num_sentences):
            txt_tgt = tgt[curr_pos:curr_pos + txt_len]
            txt_probs = probs[curr_pos:curr_pos + txt_len]
            curr_pos += txt_len
            # print(txt_probs.mean(), txt_probs[txt_tgt == 1])
            sent_ids = torch.argsort(txt_probs, descending=True)
            top_ids = sent_ids[:3]
            top_ids = top_ids.sort().values
            tgt_ids = torch.where(txt_tgt == 1)[0]
            # print(top_ids, tgt_ids)

            oracle_ids = [i for i in loader.dataset.items[iis[txt_i]]['oracle'] if i < txt_len]
            assert (torch.tensor(oracle_ids) == tgt_ids).all()

            text = loader.dataset.items[iis[txt_i]]['text']
            hyp = ' '.join(text[j] for j in sorted(top_ids))
            ref = ' '.join(loader.dataset.items[iis[txt_i]]['summary'])
            rouges.append(calc_rouge(hyp, ref))

            top_ids = [int(_) for _ in top_ids]
            tgt_ids = [int(_) for _ in tgt_ids]
            ious.append(len(set(tgt_ids).intersection(set(top_ids))) / len(set(tgt_ids).union(set(top_ids))))

    print(f'Mean loss: {np.mean(losses).item():0.03f}')
    print(f'Mean ious: {np.mean(ious).item():0.03f}')
    mean_rouge = calc_mean_rouge(rouges)
    pprint(mean_rouge)


def train(
        seed=None,
        device=None,
        batch_size=None,
        grad_accum_steps=None,
        num_steps_total=None,
        num_steps_checkpoint=None,
        num_steps_warmup_e=None,
        lr_e=None,
        num_steps_warmup_d=None,
        lr_d=None,
        wd_e=None,
        wd_d=None,
        num_workers=None,
        data_path=None,
):
    if seed is not None:
        set_seed(123)

    set_device(device)

    model = BertSumExt(
        finetune_bert=True,
    ).to(get_device())

    train_loader, test_loader = BertSumExtDataset.load_data_gazeta(
        data_path, batch_size, model.tokenizer, model.bert.config.max_position_embeddings, num_workers,
    )

    optimizer_e = AdamW(model.bert.parameters(), lr=lr_e, weight_decay=wd_e)
    optimizer_e.zero_grad()
    scheduler_e = get_linear_schedule_with_warmup(
        optimizer_e, num_warmup_steps=num_steps_warmup_e, num_training_steps=num_steps_total)

    optimizer_d = AdamW(model.classifier.parameters(), lr=lr_d, weight_decay=wd_d)
    optimizer_d.zero_grad()
    scheduler_d = get_linear_schedule_with_warmup(
        optimizer_d, num_warmup_steps=num_steps_warmup_d, num_training_steps=num_steps_total)

    step_counter = 0
    train_losses = []
    for epoch in range(100000):
        print(f'Epoch {epoch}')
        train_iter = tqdm(train_loader)
        for iis, inp, tgt in train_iter:
            step_counter += 1
            model.train()
            out, loss = model(inp.to(get_device()), tgt.to(get_device()))
            batch_loss = loss.item()
            loss = loss / grad_accum_steps  # Normalize loss (if averaged) https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
            loss.backward()
            if step_counter % grad_accum_steps == 0:
                optimizer_e.step()
                scheduler_e.step()
                optimizer_e.zero_grad()
                optimizer_d.step()
                scheduler_d.step()
                optimizer_d.zero_grad()
                assert all(lr >= 0 for lr in iter_chain(scheduler_e.get_last_lr(), scheduler_d.get_last_lr()))

            train_iter.set_description_str(f'Train loss {batch_loss:.03} ')
            train_losses.append(batch_loss)

            if step_counter % num_steps_checkpoint == 0 or step_counter >= num_steps_total:
                print(f'Step {step_counter}. Train loss {batch_loss:.03} ')
                torch.save(model.state_dict(), os.path.join(data_path, 'rus', f'bertsumext_{step_counter}.pth'))
                if step_counter >= num_steps_total:
                    return (
                        model,
                        train_losses,
                        train_loader,
                        test_loader,
                    )




