#!/usr/bin/env bash

# before:
#
# apt-get update
# apt-get install git unzip wget vim -y --fix-missing
# pip install matplotlib transformers rouge pprint tqdm
#
# git config --global user.email "alexanderkazak@yandex.ru"
# git config --global user.name "AlexanderKazakov"
# git clone https://AlexanderKazakov:password@github.com/AlexanderKazakov/summarization_nn.git

cd summarization_nn
chmod +x setup.sh
chmod +x train.sh

mkdir -p data/ckpts
cd data
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12jm_uRXynQxmRrgteWXArqcqQYm5jHJK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12jm_uRXynQxmRrgteWXArqcqQYm5jHJK" -O lenta-ru-news.zip
rm cookies.txt
unzip lenta-ru-news.zip
rm lenta-ru-news.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Hn7Ie1HMEwowLpOdYtMXuWLfq0l2bpmB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Hn7Ie1HMEwowLpOdYtMXuWLfq0l2bpmB" -O sportsru.zip && rm -rf /tmp/cookies.txt
unzip sportsru.zip
rm sportsru.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14Knam93jQE7Gz27rXX896YHA-mWsGnxA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14Knam93jQE7Gz27rXX896YHA-mWsGnxA" -O processed-ria.zip && rm -rf /tmp/cookies.txt
unzip processed-ria.zip
rm processed-ria.zip

cd ckpts
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vOMC9U3Bddgg9M-LEX12U92LNRv6h--t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vOMC9U3Bddgg9M-LEX12U92LNRv6h--t" -O rubart_initial_weights_from_rubert.zip
rm cookies.txt
unzip rubart_initial_weights_from_rubert.zip
rm rubart_initial_weights_from_rubert.zip


