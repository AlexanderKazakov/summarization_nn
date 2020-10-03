from bert_sum_ext.BertSumExt import *
from bert_sum_ext.bertsumext_data_readers import *
from bert_sum_ext.bertsumext_eval import evaluate
from transformers import AdamW, get_linear_schedule_with_warmup


def train(
        seed,
        device,
        batch_size,
        grad_accum_steps,
        num_steps_total,
        num_steps_checkpoint,
        num_steps_warmup_e,
        lr_e,
        num_steps_warmup_d,
        lr_d,
        wd_e,
        wd_d,
        num_workers,
        data_path,
        finetune_bert,
        single_batch,
        pretrained_bert_model_name,
        schedule,
        pool,
        scheduler_multiplier,
):
    set_seed(123)
    set_device(device)
    schedule = arg2bool(schedule)

    model = BertSumExt(
        pretrained_bert_model_name=pretrained_bert_model_name,
        finetune_bert=arg2bool(finetune_bert),
        pool=pool,
    ).to(get_device())

    train_loader, test_loader = BertSumExtDataset.load_data_gazeta(
        data_path,
        batch_size,
        model.tokenizer,
        model.bert.config.max_position_embeddings,
        num_workers,
        arg2bool(single_batch),
        seed,
    )

    optimizer_e = AdamW(model.bert.parameters(), lr=lr_e, weight_decay=wd_e)
    optimizer_e.zero_grad()
    if schedule:
        scheduler_e = get_linear_schedule_with_warmup(
            optimizer_e, num_warmup_steps=num_steps_warmup_e, num_training_steps=num_steps_total * scheduler_multiplier)

    optimizer_d = AdamW(model.classifier.parameters(), lr=lr_d, weight_decay=wd_d)
    optimizer_d.zero_grad()
    if schedule:
        scheduler_d = get_linear_schedule_with_warmup(
            optimizer_d, num_warmup_steps=num_steps_warmup_d, num_training_steps=num_steps_total * scheduler_multiplier)

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
                if schedule:
                    scheduler_e.step()
                optimizer_e.zero_grad()
                optimizer_d.step()
                if schedule:
                    scheduler_d.step()
                optimizer_d.zero_grad()

            train_iter.set_description_str(f'Train loss {batch_loss:.03} ')
            train_losses.append(batch_loss)

            if step_counter % num_steps_checkpoint == 0 or step_counter >= num_steps_total:
                print(f'Step {step_counter}. Train loss {batch_loss:.03} ')
                torch.save(model.state_dict(), os.path.join(data_path, f'bertsumext_{step_counter}.pth'))
                mean_test_loss, mean_iou, mean_rouge, model_hist, target_hist = evaluate(
                    model, test_loader, top_n=3, verbose=False, lowercase_rouge=False)
                print(f'Mean test loss: {mean_test_loss:0.03f}')
                print(f'Mean iou: {mean_iou:0.03f}')
                pprint(mean_rouge)

                with open(os.path.join(data_path, f'bertsumext_{step_counter}.json'), 'w', encoding='utf-8') as jf:
                    json.dump({
                        'loss': mean_test_loss,
                        'iou': mean_iou,
                        'rouges': mean_rouge,
                        'model_hist': model_hist.tolist(),
                        'target_hist': target_hist.tolist(),
                    }, jf)

                if step_counter >= num_steps_total:
                    train_iter.close()
                    return (
                        model,
                        train_losses,
                        train_loader,
                        test_loader,
                    )




