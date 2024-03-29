import sys
import os

sys.path.insert(0, os.getcwd())
from utils.data_utils import *
from rubart.common import *
from rubart.modeling_rubart import BartForConditionalGeneration


# CKPT_PATH = 'saved_models/lenta_pretrained'
CKPT_PATH = 'saved_models/gazeta_trained'
set_device('cpu')
set_seed(123)
set_max_len_src(512)
set_max_len_tgt(128)
set_min_len_tgt(1)

_, tokenizer = load_rubart_with_pretrained_encoder()
model = BartForConditionalGeneration.from_pretrained(CKPT_PATH).eval()

train_loader, test_loader = read_dataset_to_loaders(
    'gazeta', 1, CollateFnStart(tokenizer, get_max_len_src(), get_max_len_tgt()), get_max_len_src(), get_max_len_tgt())

model.eval()
for text, title in test_loader.dataset:
    source_ids = encode_text(tokenizer, text, get_max_len_src())
    encoder_attention_mask = source_ids != tokenizer.pad_token_id
    generated_ids = model.generate(
        input_ids=source_ids,
        attention_mask=encoder_attention_mask,
        num_beams=4,
        min_length=get_min_len_tgt(),
        max_length=get_max_len_tgt(),
    )

    generated_title = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    pprint([text, len(text.split())])
    decoded_text = tokenizer.decode(source_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    pprint(decoded_text)
    print('-' * 50)
    print(title)
    print('-' * 50)
    print(generated_title)
    print('=' * 50 + '\n')




