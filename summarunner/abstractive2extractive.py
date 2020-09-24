from utils.data_utils import *
import traceback
import json
from itertools import combinations, chain

# https://github.com/IlyaGusev/summarus/blob/master/summarus/util/build_oracle.py


def build_oracle_summary_greedy(text, summary, text_lemm, summary_lemm, calc_score, beam_depth):
    output = {
        "text": text,
        "summary": summary,
        "text_lemm": text_lemm,
        "summary_lemm": summary_lemm,
    }
    first_summary_sentence_lemm = summary_lemm[0]
    summary_lemm = ' '.join(summary_lemm)

    def indices_to_text(indices):
        return " ".join([text_lemm[index] for index in sorted(list(indices))])

    def calc_next_best_score(reference_summary):
        scores = []
        candidates_ids = set(range(len(text_lemm))).difference(final_indices)
        candidates_ids = chain(*[combinations(candidates_ids, n_sents) for n_sents in range(1, beam_depth + 1)])
        for ids in candidates_ids:
            test_indices = final_indices.union(set(ids))
            test_summary = indices_to_text(test_indices)
            scores.append((calc_score(test_summary, reference_summary), test_indices))
        return max(scores)

    final_indices = set()
    # heuristic: the first sentence in the summary is more important,
    # so choose the first oracle sentence to be the closest match to it
    _, final_indices = calc_next_best_score(first_summary_sentence_lemm)

    final_score = -1.0
    for _ in range(len(text_lemm)):
        best_score, best_indices = calc_next_best_score(summary_lemm)
        if best_score > final_score:
            final_score, final_indices = best_score, best_indices
        else:
            break

    output.update({
        "oracle": sorted(final_indices),
        "oracle_score": final_score
    })
    return output


def calc_single_score(pred_summary, gold_summary):
    score = calc_rouge(pred_summary, gold_summary)
    return (score['rouge-2']['f'] + score['rouge-1']['f'] + score['rouge-l']['f']) / 3


if __name__ == '__main__':
    in_file_name = os.path.join(DATA_PATH, 'rus', 'gazeta_lemmatized.jsonl')
    num_records = get_num_lines_in_file(in_file_name, encoding='utf-8')
    in_file = open(in_file_name, 'r', encoding='utf-8')
    out_file_name = os.path.join(DATA_PATH, 'rus', 'gazeta_for_summarunner.jsonl')
    out_file = open(out_file_name, 'w', encoding='utf-8')
    max_records_to_handle = 1000000000000
    counter = 0
    for input_line in tqdm(in_file, total=num_records):
        counter += 1
        if counter > max_records_to_handle:
            break

        input_dict = json.loads(input_line)
        text, summ, text_lemm, summ_lemm = input_dict['text'], input_dict['summ'], input_dict['text_lemm'], input_dict['summ_lemm']
        try:
            res = build_oracle_summary_greedy(
                text, summ, text_lemm, summ_lemm, calc_single_score, beam_depth=1
            )

            # ensure_ascii=False to prevent hieroglyphs on russian letters
            out_file.write(json.dumps(res, ensure_ascii=False) + '\n')

            # print('-' * 50)
            # pprint(res['summary'])
            # indices = res['oracle']
            # print(indices, res['oracle_score'])
            # extr_sum = [res['text'][i] for i in indices]
            # pprint(' '.join(extr_sum))

        except Exception as e:
            print('ERROR')
            print(traceback.format_exc())
            print()

    in_file.close()
    out_file.close()

    with open(out_file_name, 'r', encoding='utf-8') as out_file:
        items = out_file.readlines()
        items = random.sample(items, min(len(items), 5))
        items = [json.loads(item) for item in items]

    for item in items:
        pprint(item)










