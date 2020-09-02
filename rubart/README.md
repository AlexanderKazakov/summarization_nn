# RuBART

This work adopts [BART](https://arxiv.org/abs/1910.13461) architecture to the 
task of abstractive summarization of russian texts.

The only publicly available transformer models pretrained on russian texts (on 01.09.2020)
are RuBERT models by [DeepPalov](http://docs.deeppavlov.ai/en/master/index.html).
These models are available as part of [Huggingface transformers library](https://huggingface.co/).

The idea of that work is to slightly modify the [original BART architecture](https://huggingface.co/transformers/model_doc/bart.html) 
in order to be able to reuse weights of [DeepPavlov's RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased) 
in the embeddings and encoder layers of BART.

Modifications include(see load_rubert_to_bart_encoder.py, modeling_rubart.py):
- Changing some dimensions and parameters of the model (hidden size, etc)
- Incorporating token-type embeddings (dummy all-zeros) into BART model
- Manual weights binding, as codes of these two models are slightly different

Embeddings are shared between encoder, decoder and output classifier. 
As embeddings are from the pretrained model, the tokenizer (WordPiece) is also borrowed from 
the DeepPavlov's RuBERT model.

So, the only initially unoptimized parameters of the model are decoder layers.

### Training
The model ```saved_models/lenta_pretrained``` was trained on Lenta.Ru-News-Dataset. 

Tokenized input texts were truncated by max length 256, tokenized titles -- by max length 24.

Firstly, just model decoder's layers were trained for 3 epochs while other weights are frozen.
Then, all model's weights were trained for 5 epochs with reduced learning rate.

### Results
Results were obtained using top-4 beam-search decoding
####Lenta.Ru-News-Dataset
The average rouge results on Lenta dataset (validation part):

| r1-f | r1-p | r1-r | r2-f | r2-p | r2-r | rl-f | rl-p | rl-f | (r1-f + r2-f + rl-f) / 3 |
|------|------|------|------|------|------|------|------|------|--------------------------|
| 37.7 | 39.4 | 37.3 | 21.1 | 22.1 | 20.9 | 36.6 | 37.9 | 35.7 | 31.8                     |

Results do not show improvements over [Headline Generation Shared Task on Dialogue’2019](http://www.dialog-21.ru/media/4661/camerareadysubmission-157.pdf). 
However, manual check of the results on the validation set show reasonable quality of summarization:

| orig/pred | Title  |
|-----------|-----------------------------------------------------------------|
|  -------  | --------------------------------------------------------------- |
| orig      | США проследят за выборами президента Тайваня с двух авианосцев  |
| pred      | США направили к берегам Тайваня два авианосца                   |
|  -------  | --------------------------------------------------------------- |
| orig      | Французскую судью с 2000 года заставляли подсуживать канадцам   |
| pred      | Судья по фигурному катанию рассказала о давлении на судей в США |
|  -------  | --------------------------------------------------------------- |
| orig      | Два смертных приговора в США исполнили в один день впервые за 17 лет   |
| pred      | В США впервые с 2000 года казнили двух заключенных |
|  -------  | --------------------------------------------------------------- |
| orig      | Бекмамбетов доверит спасение Москвы американской туристке  |
| pred      | У Тимура Бекмамбетова появится актриса из " самого темного часа " |
|  -------  | --------------------------------------------------------------- |
| orig      | МИД Украины разрешил Лимонову съездить в Харьков  |
| pred      | Лимонову разрешили вернуться на Украину |
|  -------  | --------------------------------------------------------------- |
| orig      | Преступник запрыгнул в клетку ко львам и отделался укусом пальца  |
| pred      | Беглый преступник прыгнул в вольер ко львам и отделался укушенным пальцем |


