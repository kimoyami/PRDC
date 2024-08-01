from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, Dict, Tuple, Union
from simulator.model.model_base import model_base, mlp, CosineSimilarityLoss

import torch
import torch.nn as nn
import tqdm
from utils.buffer import ReplayBuffer


class model_GAN(nn.Module, model_base):
    def __init__(self) -> None:
        super.__init__()

    def train(self, replay_buffer):
        pass
