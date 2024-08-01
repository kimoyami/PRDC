from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, Dict, Tuple, Union
from simulator.model.model_base import model_base, mlp, CosineSimilarityLoss

import torch
import torch.nn as nn
import tqdm
from utils.buffer import ReplayBuffer

@dataclass
class MdlOptPair:
    net: nn.Module
    loss_f: nn.modules.loss._Loss
    optimizer: torch.optim.optimizer = None

class model_MMLP(nn.Module, model_base):
    """
    MultiMLP simulator. Seperately train the MLP to predict the next state, reward and is_done
    """

    def __init__(self, device, mlp_shape, lr=0.01, batch_size=256, **kwargs) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.device = torch.device(device)
        
        mlp_shape_next = deepcopy(mlp_shape)
        mlp_shape_next[-1] -= 2 # remove reward and done
        mlp_shape_rew = deepcopy(mlp_shape)
        mlp_shape_rew[-1] = 1 # only reward
        mlp_shape_done = deepcopy(mlp_shape)
        mlp_shape_done[-1] = 1 # only done

        self.model_next = mlp(mlp_shape_next, nn.ReLU)
        self.model_rew = mlp(mlp_shape_rew, nn.ReLU)
        self.model_done = mlp(mlp_shape_done, nn.ReLU)
        
        self.mdls: Dict[str, MdlOptPair] = {
            "next": MdlOptPair(self.model_next, CosineSimilarityLoss()),
            "rew":  MdlOptPair(self.model_rew , nn.MSELoss()),
            "done": MdlOptPair(self.model_done, nn.MSELoss())
        }

        for key, pair in self.mdls.items():
            pair.net.to(self.device)
            pair.optimizer = torch.optim.Adam(pair.net.parameters(), lr=lr)
            
    def forward(self, input:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # do not clip output
        return self.mdls["next"](input), self.mdls["rew"](input), self.mdls["done"](input)

    def train(self, replay_buffer:ReplayBuffer, epoches=1e3, 
                eval_round:int=1000, eval_func:Callable[[model_base, int], Union[dict, str]]=None,
                save_round:int=1000, save_func:Callable[[model_base, int], Union[dict, str]]=None,
                **kwargs):
        eval_round = int(eval_round)
        save_round = int(save_round)
        # norm_func = torch.tanh
        pbar = tqdm.tqdm(range(int(epoches)))
        pbar.set_description("Train MLP....")
        for epoch in pbar:
            state, action, reward, next_state, not_done = replay_buffer.sample(self.batch_size)

            cur_real = torch.cat([state, action], 1)
            for key, pair in self.mdls.items():
                pair.optimizer.zero_grad()
            p_next, p_rew, p_done = self.forward(cur_real)

            self.mdls["next"].loss_f(next_state, p_next)
            self.mdls["red" ].loss_f(reward, p_rew)
            self.mdls["done"].loss_f(not_done, p_done)

            for key, pair in self.mdls.items():
                pair.optimizer.step()

            if epoch % eval_round == 0:
                # default eval method
                if eval_func is not None:
                    msg = eval_func(self, epoch, loss=0)
                    # print(msg)
                pass

            if epoch % save_round == 0:
                # default save method
                if save_func is not None:
                    msg = save_func(self, epoch)
                    # print(msg)
                pass
            pass
        
        print("Train finished.")