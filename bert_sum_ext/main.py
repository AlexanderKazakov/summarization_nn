import sys
import os
sys.path.append(os.getcwd())

from bert_sum_ext.bertsumext_train import *


if __name__ == '__main__':
    train_args = {
        'seed': 123,
        'device': 'cuda',
        'batch_size': 4,
        'grad_accum_steps': 2,
        'num_steps_total': 1000,
        'num_steps_checkpoint': 2,
        'num_steps_warmup_e': 200,
        'lr_e': 2e-5,
        'num_steps_warmup_d': 100,
        'lr_d': 5e-5,
        'wd_e': 0.0,
        'wd_d': 0.0,
        'num_workers': 0,
        'data_path': 'data',
        'finetune_bert': 0,
        'single_batch': 1,
        # 'pretrained_bert_model_name': 'DeepPavlov/rubert-base-cased',
        'pretrained_bert_model_name': 'DeepPavlov/rubert-base-cased-sentence',
        'schedule': 1,
        'pool': 'avg',
        'scheduler_multiplier': 1.25,
        'do_basic_tokenize': False,
    }
    # TODO max_grad_norm, zero wd for layernorm

    parser = argparse.ArgumentParser()
    for k, v in train_args.items():
        parser.add_argument(f'--{k}', default=v)

    args = parser.parse_args()
    model, train_losses, train_loader, test_loader = train(**vars(args))
    plt.plot(train_losses)
    plt.show()




