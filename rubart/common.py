from utils.common import *


def set_max_len_src(max_len_src):
    global MAX_LEN_SRC
    MAX_LEN_SRC = max_len_src

def get_max_len_src():
    global MAX_LEN_SRC
    return MAX_LEN_SRC

def set_max_len_tgt(max_len_tgt):
    global MAX_LEN_TGT
    MAX_LEN_TGT = max_len_tgt

def get_max_len_tgt():
    global MAX_LEN_TGT
    return MAX_LEN_TGT

def set_min_len_tgt(min_len_tgt):
    global MIN_LEN_TGT
    MIN_LEN_TGT = min_len_tgt

def get_min_len_tgt():
    global MIN_LEN_TGT
    return MIN_LEN_TGT


# logging.basicConfig(level=logging.INFO)

RUBART_ENCODER_WEIGHTS_DIR = 'saved_models/rubart_initial_weights_from_rubert/'

# for lenta data optimal is 512 / 24, for ria data -- 1024 / 24, for sportsru -- 4000 / 200
MAX_LEN_SRC = None
MAX_LEN_TGT = None
MIN_LEN_TGT = None


def load_rubart_with_pretrained_encoder():
    from rubart.modeling_rubart import BartForConditionalGeneration

    tokenizer = BertTokenizer.from_pretrained(RUBART_ENCODER_WEIGHTS_DIR, do_lower_case=False)  # do_lower_case=False is crucial
    config = BartConfig.from_pretrained(RUBART_ENCODER_WEIGHTS_DIR)
    config.extra_pos_embeddings = 2
    config.task_specific_params = None
    config.min_length, config.max_length = get_min_len_tgt(), get_max_len_tgt()
    print(config)

    model = BartForConditionalGeneration(config)
    model.model.encoder.load_state_dict(torch.load(os.path.join(RUBART_ENCODER_WEIGHTS_DIR, 'encoder_state_dict.pth')))
    # embeddings sharing
    model.model.decoder.embed_positions.weight = model.model.encoder.embed_positions.weight
    model.model.decoder.token_type_embeddings.weight = model.model.encoder.token_type_embeddings.weight
    model.model.decoder.layernorm_embedding.weight = model.model.encoder.layernorm_embedding.weight
    model.model.decoder.layernorm_embedding.bias = model.model.encoder.layernorm_embedding.bias
    assert (model.model.shared.weight == model.model.encoder.embed_tokens.weight).all()
    assert (model.model.shared.weight == model.model.decoder.embed_tokens.weight).all()
    assert (model.model.encoder.embed_positions.weight == model.model.decoder.embed_positions.weight).all()
    assert (model.model.encoder.token_type_embeddings.weight == model.model.decoder.token_type_embeddings.weight).all()
    assert (model.model.encoder.layernorm_embedding.weight == model.model.decoder.layernorm_embedding.weight).all()
    assert (model.model.encoder.layernorm_embedding.bias == model.model.decoder.layernorm_embedding.bias).all()

    # the only not pretrained parameters are decoder.layers
    return model, tokenizer







