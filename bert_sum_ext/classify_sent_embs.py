from utils.common import *
from sklearn.manifold import TSNE


with open(os.path.join(DATA_PATH, 'sents_embs_3.json'), 'r') as f:
    d = json.load(f)

inp = np.array(d['sents_embs'])
tgt = d['target']

# tsne_inp = TSNE(n_components=2, perplexity=5).fit_transform(inp)
# inp = tsne_inp
#
# color = ['r' if t > 0 else 'b' for t in tgt]
# plt.scatter(inp[:, 0], inp[:, 1], c=color)
# plt.show()

inp = torch.tensor(inp).float()
tgt = torch.tensor(tgt)

model = nn.Sequential(
    nn.BatchNorm1d(inp.shape[1]),
    nn.Linear(inp.shape[1], 1),
)
model.train()
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

for iter in range(10000):
    opt.zero_grad()
    out = model(inp).squeeze()
    loss = loss_fn(out, tgt.float())
    res = (out > 0).int()
    acc = (res == tgt).sum().item() / len(tgt)
    print(f'Loss {loss.item():.03f} Acc {acc:.03f}')
    loss.backward()
    opt.step()




