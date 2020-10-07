from bert_sum_ext.bertsumext_eval import *
from torch import jit


set_device('cpu')
data_path = 'data'
# model_file, do_basic_tokenize = 'bertsumext_40000_30_09', True
model_file, do_basic_tokenize = 'bertsumext_40000_07_10', False
ckpt_path = os.path.join(data_path, 'rus', 'gazeta', model_file + '.{}')
pretrained_bert_model_name = 'DeepPavlov/rubert-base-cased-sentence'

model = BertSumExt(
    pretrained_bert_model_name=pretrained_bert_model_name,
    do_basic_tokenize=do_basic_tokenize,
    finetune_bert=False,
    pool='avg',  # TODO configs!!!
)
model.to('cpu')  # to avoid cuda out of memory while loading
model.load_state_dict(torch.load(ckpt_path.format('pth')))
model.to(get_device())
model.eval()

tokenizer = model.tokenizer
collator = BertSumExtCollateFn(
    tokenizer,
    model.sentence_len,
)

fname = '006 email bank cards.txt'
infer_dir = os.path.join(data_path, 'rus/my_inputs')
with open(os.path.join(infer_dir, fname), 'r', encoding='utf-8') as f:
    text = f.read()
_, sample, _ = collator([(0, sentenize_with_newlines(text), [])])

with torch.no_grad():
    out_1 = model(sample)[0]
    out_2 = model(sample)[0]
    assert (out_1 == out_2).all()

# see tracer warnings! sample timestamps must be always 512 because of tracing
with torch.jit.optimized_execution(True):
    traced = jit.trace(model, sample)

traced_model_path = ckpt_path.format('torchscript')
traced.save(traced_model_path)
traced = torch.jit.load(traced_model_path)
traced.eval()

with torch.no_grad():
    fnames = next(os.walk(infer_dir))[2]
    # fnames = ['3.txt']
    for fname in fnames:
        print(fname)
        with open(os.path.join(infer_dir, fname), 'r', encoding='utf-8') as f:
            text = f.read()
        _, sample, _ = collator([(0, sentenize_with_newlines(text), [])])
        sample_copy = sample.clone()
        out_py = model(sample)[0]
        assert (sample == sample_copy).all()
        out_jit = traced(sample)[0]
        assert (sample == sample_copy).all()
        out_jit_2 = traced(sample)[0]
        assert (out_jit_2 == out_jit).all()
        print(sample.shape, '->', out_jit.shape)
        # not equal! https://discuss.pytorch.org/t/will-torch-jit-script-change-the-output-of-model/69452
        assert torch.abs(out_py - out_jit).max() < 1e-4


