#!/usr/bin/env bash

# train decoder layers
python summarization/summarization_train.py --dataset=lenta --device=cuda --seed=123 --max_source_length=256 --max_target_length=24 --min_target_length=1 \
--batch_size=16 --lr=1e-4 --train_whole_model=False --scheduler_num_epochs=1 --scheduler=plateau_decay

# train whole model on lenta
python rubart/summarization_train.py --device=cuda --seed=123 --max_source_length=256 --max_target_length=24 --min_target_length=1 --batch_size=16 --init_ckpt_dir=saved_models/with_trained_decoder --lr=1e-5 --train_whole_model=True --scheduler_num_epochs=1 --scheduler=plateau_decay

# train whole model on sportsru
python rubart/summarization_train.py --dataset=sportsru --device=cuda --seed=123 --max_source_length=256 --max_target_length=256 --min_target_length=64 --batch_size=4 --init_ckpt_dir=saved_models/lenta_pretrained --lr=1e-5 --train_whole_model=True --scheduler_num_epochs=1 --scheduler=plateau_decay

# train whole model on ria
python rubart/summarization_train.py --dataset=ria --device=cuda --seed=123 --max_source_length=512 --max_target_length=24 --min_target_length=1 --batch_size=4 --init_ckpt_dir=saved_models/lenta_pretrained --lr=1e-5 --train_whole_model=True --scheduler_num_epochs=1 --scheduler=plateau_decay

# train whole model on gazeta
python rubart/summarization_train.py --dataset=gazeta --device=cuda --seed=123 --max_source_length=512 --max_target_length=84 --min_target_length=16 --batch_size=4 --init_ckpt_dir=saved_models/lenta_pretrained --lr=1e-5 --train_whole_model=True --scheduler_num_epochs=1 --scheduler=plateau_decay

# tmux:
# 'Ctrl+B' then ':new' -- new session
# ./train.sh
# 'Ctrl+B' then ')' -- next session
# 'Ctrl+B' then 'd'  -- detach without stopping process

# run in background:
# ./train.sh > /dev/null 2>&1 &
#

# ssh connection:
# ssh -i _PATH_TO_KEY_ -p _PORT_ root@_ADDRESS_ -L 8080:localhost:8080

# scp file from host to local (note big -P):
# scp -i _PATH_TO_KEY_ -P _PORT_ root@_ADDRESS_:_FILE_PATH_ON_HOST_ _LOCAL_FILE_PATH_

# make tar from ckpt:
# tar -zcvf best_ckpt.tar.gz best_ckpt/
# extract:
# tar -zxvf best_ckpt.tar.gz

