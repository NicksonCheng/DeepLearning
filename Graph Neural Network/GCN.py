import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid


dataset = pd.read_csv(
    "cora/cora.cites", sep="\t", header=None, names=["target", "source"]
)
dataset["label"] = "cites"

print(dataset.sample(frac=1).head(5))
