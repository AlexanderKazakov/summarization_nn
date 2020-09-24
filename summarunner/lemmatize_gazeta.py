from utils.data_utils import *


# set_max_num_samples(1000)
texts, summs = read_data_gazeta()
assert len(texts) == len(summs)

texts_lemm, texts = lemmatize_texts_rus(texts)
summs_lemm, summs = lemmatize_texts_rus(summs)

out_file_name = os.path.join(DATA_PATH, 'rus', 'gazeta_lemmatized.jsonl')
with open(out_file_name, 'w', encoding='utf-8') as out_file:
    for (text, summ, text_lemm, summ_lemm) in zip(texts, summs, texts_lemm, summs_lemm):
        out_file.write(json.dumps({
            'text': text,
            'text_lemm': text_lemm,
            'summ': summ,
            'summ_lemm': summ_lemm,
        }, ensure_ascii=False) + '\n')

with open(out_file_name, 'r', encoding='utf-8') as out_file:
    items = out_file.readlines()
    items = random.sample(items, min(len(items), 5))
    items = [json.loads(item) for item in items]

for item in items:
    pprint(item)


