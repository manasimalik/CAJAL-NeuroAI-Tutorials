from torch.utils.data import Sampler
import numpy as np

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm.notebook import tqdm
from neuralpredictors.data.datasets import FileTreeDataset
from neuralpredictors.measures.modules import PoissonLoss
from torch.utils.data import DataLoader



class SubsetSampler(Sampler):

    def __init__(self, indices, num_samples=None, shuffle=True):
        # If indices is a boolean array, convert it to an index array
        if np.issubdtype(indices.dtype, np.bool_):
            indices = np.nonzero(indices)[0]

        self.indices = indices
        if num_samples is None:
            num_samples = len(indices)
            
        self.num_samples = num_samples
        self.replace = num_samples > len(indices)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = np.random.choice(self.indices, size=self.num_samples, replace=self.replace)
        else:
            assert self.num_samples == len(self.indices), "Number of samples must be equal to the number of indices for non-shuffled sampling"
            indices = self.indices
        return iter(indices.tolist())
    
    def __repr__(self):
        return f"Random Subset Sampler on an array of {len(self.indices)}, {self.num_samples} samples per iteration and replace={self.replace}"

    def __len__(self):
        return self.num_samples
    
    
def corr(y1, y2, dim=-1, eps=1e-12, **kwargs):
    y1 = (y1 - y1.mean(axis=dim, keepdims=True)) / (y1.std(axis=dim, keepdims=True) + eps)
    y2 = (y2 - y2.mean(axis=dim, keepdims=True)) / (y2.std(axis=dim, keepdims=True) + eps)
    return (y1 * y2).mean(axis=dim, **kwargs)



def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for images, responses in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, responses)
        loss.backward()
        optimizer.step()
    return loss

def get_correlations(model, loader):
    """
    Calculates the correlation between the model's predictions and the actual responses.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The data loader containing the images and responses.

    Returns:
        float: The correlation between the model's predictions and the actual responses.
    """
    resp, pred = [], []
    model.eval()
    for images, responses in loader:
        outputs = model(images)
        resp.append(responses.cpu().detach().numpy())
        pred.append(outputs.cpu().detach().numpy())
    resp = np.vstack(resp)
    pred = np.vstack(pred)
    return corr(resp, pred, dim=0)