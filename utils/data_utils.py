from utils.common import *


def preprocess(texts, summs, max_len_src, max_len_tgt):
    etx = chr(3)

    def deduplicate_newlines(txt):
        txt = re.sub(etx + '{2,}', etx, txt)
        txt = re.sub('^' + etx, '', txt)
        txt = re.sub(etx + '$', '', txt)
        return txt

    def _preprocess(txt, max_len):
        txt = txt.replace('\n', etx)
        txt = deduplicate_newlines(txt)
        txt = txt.split()
        if max_len is not None:
            txt = txt[:max_len]
        txt = ' '.join(txt)
        txt = txt.replace(etx, '\n')
        return txt

    texts_, summs_ = [], []
    bad_records_counter = 0
    for text, summ in zip(texts, summs):
        text = _preprocess(text, max_len_src)
        summ = _preprocess(summ, max_len_tgt)
        if len(summ) != 0 and len(text) > len(summ) / 2:
            texts_.append(text)
            summs_.append(summ)
        else:
            bad_records_counter += int(len(summ) != 0)

    print(f'Num records: {len(texts_)}, num bad records: {bad_records_counter}')
    return texts_, summs_


def read_data_lenta(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'rus/lenta/lenta-ru-news.csv'
    print('Lenta dataset')
    texts, summs = [], []
    with open(path, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for counter, (url, title, text, topic, tags, date) in enumerate(reader):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            texts.append(text)
            summs.append(title)

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_ria(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'rus/ria/processed-ria.json'
    print('RIA dataset')
    texts, summs = [], []
    with open(path, encoding='utf8') as f:
        for counter, line in enumerate(f):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            data = json.loads(line)
            text = re.sub('<[^>]+>', '', data['text']).replace('\n', ' ').strip()
            title = re.sub('<[^>]+>', '', data['title']).replace('\n', ' ').strip()
            texts.append(text)
            summs.append(title)

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_gazeta_initial_splits():
    path = DATA_PATH + 'rus/gazeta/'
    print('Gazeta dataset')
    res = dict()
    for split in ['train', 'val', 'test']:
        res[split] = {'src': [], 'tgt': []}
        with open(path + f'gazeta_{split}.jsonl', encoding='utf8') as f:
            for counter, line in enumerate(f):
                if get_max_num_samples() is not None and counter >= get_max_num_samples() // 3:
                    break

                data = json.loads(line)
                res[split]['src'].append(data['text'])
                res[split]['tgt'].append(data['summary'])

    return res


def read_data_gazeta(max_len_src=None, max_len_tgt=None):
    texts, summs = [], []
    data = read_data_gazeta_initial_splits()
    for k, d in data.items():
        texts.extend(d['src'])
        summs.extend(d['tgt'])
    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_rus_sci_articles(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'rus/rus_sci_articles/lda_train.csv'
    print('Rus_sci_articles dataset')
    texts, summs = [], []
    with open(path, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)
        for counter, (id, title, text, summary) in enumerate(reader):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            texts.append(text)
            summs.append(summary)

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_wikihow_sep(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'eng/wikihow/wikihowSep.csv'
    print('WikihowSep dataset')
    texts, summs = [], []
    with open(path, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for counter, items in enumerate(reader):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            try:
                overview, headline, text, sectionLabel, title = items
                texts.append(text)
                summs.append(headline)
            except ValueError:
                pass

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_wikihow_all(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'eng/wikihow/wikihowAll.csv'
    print('WikihowAll dataset')
    texts, summs = [], []
    with open(path, newline='', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for counter, items in enumerate(reader):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            try:
                headline, title, text = items
                texts.append(text)
                summs.append(headline)
            except ValueError:
                pass

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_kaggle_indian_news(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'eng/kaggle_indian_news/news_summary.csv'
    print('kaggle_indian_news dataset')
    texts, summs = [], []
    with open(path, newline='', encoding='ansi') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for counter, (author, date, headlines, read_more, text, ctext) in enumerate(reader):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            texts.append(ctext)
            summs.append(text)

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_stories_cnn_dailymail(path, max_len_src, max_len_tgt):
    file_names = next(os.walk(path))[2]
    texts, summs = [], []
    for counter, fn in enumerate(file_names):
        if get_max_num_samples() is not None and counter >= get_max_num_samples() // 2:
            break

        with open(os.path.join(path, fn), encoding='utf-8') as f:
            lines = f.readlines()

        text, summ = [], []
        curr_output = text
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            elif line == '@highlight':
                curr_output = summ
                continue
            else:
                if line[-1] not in string.punctuation:
                    line += '.'
                curr_output.append(line)

        texts.append(' '.join(text))
        summs.append(' '.join(summ))

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_cnn(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'eng/cnn/stories'
    print('cnn dataset')
    return read_stories_cnn_dailymail(path, max_len_src, max_len_tgt)


def read_data_dailymail(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'eng/dailymail/stories'
    print('dailymail dataset')
    return read_stories_cnn_dailymail(path, max_len_src, max_len_tgt)


def read_data_cnn_dailymail(max_len_src=None, max_len_tgt=None):
    print('cnn/dailymail dataset')
    texts_cnn, summs_cnn = read_data_cnn(max_len_src, max_len_tgt)
    texts_dailymail, summs_dailymail = read_data_dailymail(max_len_src, max_len_tgt)
    return texts_cnn + texts_dailymail, summs_cnn + summs_dailymail


def read_data_reddit_tldr(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'eng/reddit-tldr/tldr-training-data.jsonl'
    print('Reddit tldr dataset')
    texts, summs = [], []
    with open(path, encoding='utf8') as f:
        for counter, line in enumerate(f):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            data = json.loads(line)
            text = data['content'].strip()
            summ = data['summary'].strip()
            if len(re.findall(r'tl[;.-]?dr', text.lower())) == 0:
                texts.append(text)
                summs.append(summ)

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_data_webis_snippets(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'eng/webis_snippets/released_anchorcontext.json'
    print('Webis snippets dataset')
    texts, summs = [], []
    with open(path, encoding='utf8') as f:
        for counter, line in enumerate(f):
            if get_max_num_samples() is not None and counter >= get_max_num_samples():
                break

            if counter % 1000 != 0:
                continue  # this is 65 Gb file!

            data = json.loads(line)
            assert len(data) == 2
            text = data['webpage'].strip()
            summ = data['anchorcontext'].strip()
            texts.append(text)
            summs.append(summ)

    return preprocess(texts, summs, max_len_src, max_len_tgt)


def read_dataset_to_loaders(name, batch_size, collate_fn, max_len_src, max_len_tgt):
    if name == 'lenta':
        all_texts, all_titles = read_data_lenta(max_len_src, max_len_tgt)
    elif name == 'ria':
        all_texts, all_titles = read_data_ria(max_len_src, max_len_tgt)
    elif name == 'gazeta':
        all_texts, all_titles = read_data_gazeta(max_len_src, max_len_tgt)
    elif name == 'rus_sci_articles':
        all_texts, all_titles = read_data_rus_sci_articles(max_len_src, max_len_tgt)
    elif name == 'wikihowAll':
        all_texts, all_titles = read_data_wikihow_all(max_len_src, max_len_tgt)
    elif name == 'wikihowSep':
        all_texts, all_titles = read_data_wikihow_sep(max_len_src, max_len_tgt)
    elif name == 'cnn_dailymail':
        all_texts, all_titles = read_data_cnn_dailymail(max_len_src, max_len_tgt)
    elif name == 'reddit_tldr':
        all_texts, all_titles = read_data_reddit_tldr(max_len_src, max_len_tgt)
    elif name == 'webis_snippets':
        all_texts, all_titles = read_data_webis_snippets(max_len_src, max_len_tgt)
    else:
        raise RuntimeError('Unknown dataset name')

    train_texts, val_texts, train_titles, val_titles = \
        train_test_split(all_texts, all_titles, test_size=0.1, shuffle=True)

    train_dataset = SummarizationDataset(train_texts, train_titles)
    val_dataset = SummarizationDataset(val_texts, val_titles)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=8)

    return train_loader, val_loader


def read_data_sportsru(max_len_src=None, max_len_tgt=None):
    path = DATA_PATH + 'rus/sportsru/'
    print('sports.ru dataset')
    data = {
        'train': {'src': [], 'tgt': []},
        'val': {'src': [], 'tgt': []},
        'test': {'src': [], 'tgt': []},
    }
    with open(path + 'train_src.broad.txt', encoding='utf8') as f:
        for line in f:
            data['train']['src'].append(line)
    with open(path + 'train_tgt.news.txt', encoding='utf8') as f:
        for line in f:
            data['train']['tgt'].append(line)
    with open(path + 'valid_src.broad.txt', encoding='utf8') as f:
        for line in f:
            data['val']['src'].append(line)
    with open(path + 'valid_tgt.news.txt', encoding='utf8') as f:
        for line in f:
            data['val']['tgt'].append(line)
    with open(path + 'test_src.broad.txt', encoding='utf8') as f:
        for line in f:
            data['test']['src'].append(line)
    with open(path + 'test_tgt.news.txt', encoding='utf8') as f:
        for line in f:
            data['test']['tgt'].append(line)

    for k, d in data.items():
        if max_len_src is not None:
            d['src'] = [' '.join(text.split()[-max_len_src:]) for text in d['src']]
        if max_len_tgt is not None:
            d['tgt'] = [' '.join(text.split()[-max_len_tgt:]) for text in d['tgt']]
    
    assert get_max_num_samples() is None, 'TODO'

    return data


def explore_tokenized(strings, tokenizer):
    enc, spl = [], []
    num_samples = min(len(strings), 2000)
    for t in random.sample(strings, num_samples):
        enc.append(encode_text(tokenizer, t, 100000).squeeze())
        spl.append(t.split())

    len_enc = [len(e) for e in enc]
    len_spl = [len(s) for s in spl]
    med_num_words = int(np.median(len_spl))
    print(f'dataset size = {len(strings)}')
    print(f'median text length = {med_num_words}')
    print(f'estimated total number of words in dataset = {med_num_words * len(strings)}')
    print(f'enc_len / split_len = {np.median([len(e) / len(s) for e, s in zip(enc, spl)])}')
    plt.clf()
    plt.hist(len_enc, bins=100)
    plt.hist(len_spl, bins=100, color='y')
    plt.show()


def explore_set(texts, summaries):
    assert len(texts) == len(summaries)
    num_samples = min(len(texts), 2000)
    ids = random.sample(range(len(texts)), num_samples)
    txt_lens = [len(texts[i].split()) for i in ids]
    sum_lens = [len(summaries[i].split()) for i in ids]

    txt_med_num_words = int(np.median(txt_lens))
    sum_med_num_words = int(np.median(sum_lens))
    print(f'num texts = {len(texts)}')
    print(f'median text length = {txt_med_num_words}')
    print(f'median summ length = {sum_med_num_words}')
    print(f'text_len / summ_len = {np.median([t / s for t, s in zip(txt_lens, sum_lens)])}')
    print()
    # plt.clf()
    # plt.hist(txt_lens, bins=100)
    # plt.hist(sum_lens, bins=100, color='y')
    # plt.show()


if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained(RUBART_ENCODER_WEIGHTS_DIR, do_lower_case=False)  # do_lower_case=False is crucial

    # texts, summs = read_data_webis_snippets()
    # explore_set(texts, summs)

    # texts, summs = read_data_reddit_tldr()
    # explore_set(texts, summs)

    # texts, summs = read_data_cnn_dailymail()
    # explore_set(texts, summs)

    # texts, summs = read_data_kaggle_indian_news()
    # explore_set(texts, summs)

    # texts, summs = read_data_wikihow_all()
    # explore_set(texts, summs)

    # texts, summs = read_data_wikihow_sep()
    # explore_set(texts, summs)

    texts, summs = read_data_gazeta()
    explore_set(texts, summs)

    # texts, summs = read_data_rus_sci_articles()
    # explore_set(texts, summs)

    # texts, summs = read_data_lenta()
    # explore_set(texts, summs)

    # texts, summs = read_data_ria()
    # explore_set(texts, summs)

    # data_sportsru = read_data_sportsru()
    # explore_set(data_sportsru['train']['src'], data_sportsru['train']['tgt'])









