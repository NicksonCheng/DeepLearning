import torch
import time
from torch import nn, Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple
from transformer import TransformerModel


train_iter = WikiText2(split="train")

## split sentance
tokenizer = get_tokenizer("basic_english")
# build vocabulary list
## special_first – Indicates whether to insert symbols at the beginning or at the end.
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def data_preprocessing(raw_text_iter) -> Tensor:
    ## conver raw text into flat Tensor
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
    ]
    ## remove zero length sentance
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


train_iter, val_iter, test_iter = WikiText2()


batch_size = 20
eval_batch_size = 10
train_data = data_preprocessing(train_iter)
val_data = data_preprocessing(val_iter)
test_data = data_preprocessing(test_iter)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data: Tensor, batch_size: int) -> Tensor:
    seq_len = data.size(0) // batch_size
    data = data[: seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data.to(device)


train_data = batchify(train_data, batch_size=batch_size)
val_data = batchify(val_data, batch_size=eval_batch_size)
test_data = batchify(test_data, batch_size=eval_batch_size)


bptt = 35  # subdivides the source data into chunks of length


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(bptt, len(source) - 1)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + seq_len + 1].reshape(-1)
    print(source.shape)
    print(data.shape)
    print(target.shape)
    return data, target


######################
## Model Initialize ##
######################


ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


model = TransformerModel(
    ntoken=ntokens,
    d_model=emsize,
    d_hid=d_hid,
    nlayers=nlayers,
    nhead=nhead,
    dropout=dropout,
)


######################
## Training Process ##
######################
epoches = 10
criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module) -> None:
    model.train()
    total = train_data.size(0)
    total_loss = 0.0
    log_val=200
    start = time.time()

    ## every time source is i and target is i+1
    for batch, i in enumerate(range(0, total, bptt)):
        data, target = get_batch(train_data, i)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters().0.5) # trimming the gradient
        optimizer.step()

        total_loss+=loss.item()

        ## record loss

        if batch

for epoch in range(epoches):
    train(model)
    break
