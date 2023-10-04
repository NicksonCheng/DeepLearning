####################################################
##
## This Model is simple import Transformer Architecture from pytorch
##
##
####################################################

import torch
import time
from torch import nn, Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple
from Transformer import TransformerModel
from tqdm import tqdm

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
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + seq_len + 1].reshape(-1)
    return data, target


######################
## Model Initialize ##
######################


ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


model = TransformerModel(
    ntoken=ntokens,
    d_model=emsize,
    d_hid=d_hid,
    nlayers=nlayers,
    nhead=nhead,
    dropout=dropout,
).to(device=device)
exit()
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
    total = train_data.size(0) - 1
    num_batches = total // bptt
    batch_total_loss = 0.0
    log_interval = 200
    start = time.time()
    ## every time source is i and target is i+1
    for batch, i in tqdm(enumerate(range(0, total, bptt))):
        data, targets = get_batch(train_data, i)
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        # print(output_flat.size(), targets.size())
        # print(output_flat[0], targets[0])
        loss = criterion(output_flat, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # trimming the gradient
        optimizer.step()

        batch_total_loss += loss.item() / log_interval
        ## record loss
        if batch % log_interval == 0 and batch > 0:
            print(
                f"epoch {epoch} | { batch } //{ num_batches } batches   \
             | loss: {batch_total_loss} | time: {time.time()-start}"
            )
            batch_total_loss = 0
            start = time.time()


def evaluate(model: nn.Module) -> None:
    model.eval()
    total = val_data.size(0) - 1
    total_loss = 0.0

    for batch, i in tqdm(enumerate(range(0, total, bptt))):
        data, targets = get_batch(val_data, i)
        seq_len = data.size(0)
        data = data.to(device)
        targets = targets.to(device)

        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        total_loss += seq_len * loss.item()
    total_loss = total_loss / total
    return total_loss


for epoch in range(epoches):
    epoch_start_time = time.time()

    train(model)
    epoch_total_val_loss = evaluate(model)
    print("-" * 100)
    print(
        f"epoch: {epoch} | val loss: {epoch_total_val_loss} | time: {time.time()-epoch_start_time}"
    )
    print("-" * 100)
