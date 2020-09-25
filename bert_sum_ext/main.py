import sys
import os
sys.path.append(os.getcwd())

from bert_sum_ext.bertsumext_train import *


if __name__ == '__main__':
    train_args = {
        'seed': 123,
        'device': 'cuda',
        'batch_size': 4,
        'grad_accum_steps': 4,
        'num_steps_total': 1000,
        'num_steps_checkpoint': 200,
        'num_steps_warmup_e': 200,
        'lr_e': 0.001,
        'num_steps_warmup_d': 100,
        'lr_d': 0.05,
        'wd_e': 0.0,
        'wd_d': 0.0,
        'num_workers': 0,
        'data_path': '../data_summarization',
    }
    # TODO max_grad_norm, zero wd for layernorm, schedule and other params from paper

    parser = argparse.ArgumentParser()
    for k, v in train_args.items():
        parser.add_argument(f'--{k}', default=v)

    args = parser.parse_args()
    model, train_losses, train_loader, test_loader = train(**vars(args))
    plt.plot(train_losses)
    plt.show()

    # model.to('cpu')  # to avoid cuda out of memory while loading
    # model.load_state_dict(torch.load(os.path.join(data_path, 'rus', '_1', 'bertsumext_1000.pth')))
    # model.to(get_device())
    # evaluate(model, test_loader)
    # return


