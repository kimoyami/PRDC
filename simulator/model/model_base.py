from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import tqdm
from typing import Callable, Dict, Literal, Tuple, Union
from abc import ABC, abstractmethod
from utils.buffer import ReplayBuffer

class model_base(ABC):

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def forward(self, input):
        ...

    # @abstractmethod
    # def step(self, state, action):
    #     ...

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return

    pass

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

"""
Two possible loss for the simulator.
"""

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output, target):
        output_norm = F.normalize(output, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        similarity = torch.sum(output_norm * target_norm, dim=-1)
        loss = 1 - similarity  # Cosine similarity loss
        return torch.mean(loss)
    
class NormLoss(nn.Module):
    def __init__(self):
        super(NormLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.norm(output - target)
        return loss

if __name__ == "__main__":
    pass