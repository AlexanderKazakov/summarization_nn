import sys
import os
sys.path.append(os.getcwd())

from bert_sum_ext.bertsumext_train import *


if __name__ == '__main__':
    train_args = {
        'seed': 123,
        'device': 'cpu',
        'batch_size': 1,
        'grad_accum_steps': 1,
        'num_steps_total': 1000,
        'num_steps_checkpoint': 100,
        'num_steps_warmup_e': 200,
        'lr_e': 2e-5,
        'num_steps_warmup_d': 100,
        'lr_d': 1e-4,
        'wd_e': 0.0,
        'wd_d': 0.0,
        'num_workers': 0,
        'data_path': '../data_summarization',
        'finetune_bert': 1,
        'single_batch': 1,
        # 'pretrained_bert_model_name': 'DeepPavlov/rubert-base-cased',
        'pretrained_bert_model_name': 'DeepPavlov/rubert-base-cased-sentence',
        'schedule': 1,
    }
    # TODO max_grad_norm, zero wd for layernorm

    parser = argparse.ArgumentParser()
    for k, v in train_args.items():
        parser.add_argument(f'--{k}', default=v)

    args = parser.parse_args()
    model, train_losses, train_loader, test_loader = train(**vars(args))
    plt.plot(train_losses)
    plt.show()




