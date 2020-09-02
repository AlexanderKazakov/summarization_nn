#Датасеты для суммаризации:
*большинство датасетов (если не все) для некоммерческого использования!*

##Русские:

1. Лента — новости, короткий текст - короткий заголовок, cased
    - Num records: 799632, num bad records: 1343
    - median text length = 170
    - median summ length = 7
    - text_len / summ_len = 23

2. РИА — новости, короткий текст - короткий заголовок, uncased
    - Num records: 1003682, num bad records: 187
    - median text length = 189
    - median summ length = 9
    - text_len / summ_len = 21

3. Газета.ру — новости, текст подлиннее и саммари (не заголовки!) подлиннее (есть также и заголовки)
    - Num records: 63434, num bad records: 1
    - median text length = 599
    - median summ length = 42
    - text_len / summ_len = 15

4. Русские научные статьи на разные темы с саммари=абстрактами (Rus_sci_articles)
    - Num records: 1214, num bad records: 0
    - median text length = 2336
    - median summ length = 52
    - text_len / summ_len = 47


##Английские:

1. Wikihow -- не новости! длинные тексты, есть как суммаризации отдельных абзацев (WikihowSep), так и склеенные из них для всего текста (WikihowAll). можно наскрапить таких же с русского сайта.
    - WikihowAll dataset 
        - Num records: 197283, num bad records: 17262
        - median text length = 338
        - median summ length = 40
        - text_len / summ_len = 10
    - WikihowSep dataset
        - Num records: 1350285, num bad records: 235211
        - median text length = 57
        - median summ length = 6
        - text_len / summ_len = 10

2. Reddit tldr dataset
    - Num records: 3084269, num bad records: 13
    - median text length = 194
    - median summ length = 19
    - text_len / summ_len = 10

3. cnn/dailymail dataset (по отдельности cnn 92K, dailymail 219K записей, параметры одинаковые) -- новости, небольшой текст в несколько предложений, саммари подлиннее, cased
    - Num records: 311959, num bad records: 126
    - median text length = 614
    - median summ length = 45
    - text_len / summ_len = 13

4. kaggle_indian_news dataset -- новости из индии: а) news_summary.csv - новости, небольшой датасет, сжатие всего в пять раз; б) news_summary_more.csv - сильное сжатие (саммари=заголовок)
    - Num records: 4367, num bad records: 147
    - median text length = 284
    - median summ length = 59
    - text_len / summ_len = 5

5. Webis snippets dataset -- огромный датасет 65 Гб, сжатие всего в два раза, качество саммари неочевидно
    - Num records: ~10M, num bad records: ~0.5M
    - median text length = 357
    - median summ length = 190
    - text_len / summ_len = 2


##Ещё (не скачивал):
- MATINF — многотасковый, в том числе суммаризация текста вопроса
- MLSUM — multilingual (including russian)
- samsum — qa-summarization dataset
- newsroom — новости, большой датасет, саммари короткие, экстр и абстр и смеш подходы. качать либо через скрапер скрипт, либо получить через соглашение по емеил
- GIGAWORD — новости, короткий текст - короткий заголовок, cased
- BBC News — extraction based
- australian legal cases — довольно мудрёная структура, набор приговоров суда, которые друг на друга ссылаются, вроде как может быть использована для экстратной суммаризации (??)
- Opinosis — нестандартный подход к суммаризации — графы плюс нечто среднее между абстрактной и экстрактной. Небольной датасет с маленькими текстами
- TALKSUMM - ещё датасет, интересный способ сбора данных
- science-summarization — https://github.com/ninikolov/data-driven-summarization -- медицинские статьи на английском. а) название по абстракту, б) абстракт по статье
- https://www.kaggle.com/Cornell-University/arxiv — полный датасет статей архив.орг с категориями
- Multi-News -- multi-document summarization
- XSum -- focuses on very abstractive summaries


