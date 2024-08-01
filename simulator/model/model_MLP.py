from typing import Callable, Literal, Dict, Union
from simulator.model.model_base import model_base, mlp, CosineSimilarityLoss

import torch
import torch.nn as nn
import tqdm
from utils.buffer import ReplayBuffer

class model_MLP(nn.Module, model_base):
    """
    A network to simulate the env model.
    """
    def __init__(self, device, mlp_shape, lr=0.01, batch_size=256, **kwargs) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.model = mlp(mlp_shape, nn.ReLU)
        
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.loss_f = nn.BCELoss()
        self.loss_f = CosineSimilarityLoss()

    def forward(self, input:torch.Tensor):
        # do not clip output
        return self.model(input)

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
            next_real = torch.cat([reward, next_state, not_done], 1)
            next_pred = self.forward(cur_real)

            loss:torch.Tensor = self.loss_f(next_real, next_pred)
            loss.backward()
            self.optimizer.step()
            _loss = loss.detach().item()
            pbar.set_description(f"loss={_loss:0.3}")
            if epoch % eval_round == 0:
                # default eval method
                if eval_func is not None:
                    msg = eval_func(self, epoch, loss=_loss)
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