import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from argparse import ArgumentParser
from functools import partial
from typing import List


class OthelloAILayer(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16, bias=True):
        super(OthelloAILayer, self).__init__()
        self.in_dims = in_dims
        self.in_proj = nn.Linear(in_dims, hidden_dims, bias)
        self.active = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_proj = nn.Linear(hidden_dims, hidden_dims)
        self.out_proj = nn.Linear(hidden_dims, out_dims)

    def __call__(self, x):
        # xの方が何かを調べる？？
        # それはあとでいいんじゃね？
        # print("#" * 10)
        # print("layer")
        # print(f"x shape ={x.shape}")
        # print(f"self.in_dims={self.in_dims}")
        x = self.in_proj(x)
        # print(f"x shape ={x.shape}")
        x = self.active(x)
        x = self.hidden_proj(x)
        x = self.active(x)
        x = self.out_proj(x)
        x = self.active(x)
        return x


class OthelloAI(nn.Module):
    def __init__(self, in_dims_list: List[int], out_dims: int, hidden_dims: int):
        super(OthelloAI, self).__init__()
        # print(f"in_dims_list={in_dims_list}")
        self.ai_layers: List[OthelloAILayer] = []
        for in_dims in in_dims_list:
            self.ai_layers.append(
                OthelloAILayer(in_dims=in_dims, out_dims=1, hidden_dims=hidden_dims)
            )
        # print(f"len in_dims_list={len(in_dims_list)}")
        self.out_proj = nn.Linear(len(in_dims_list), out_dims)

    def __call__(self, x_list):

        y_list = []
        for idx in range(len(self.ai_layers)):
            # print(f"shape x_list[{idx}]={x_list[idx].shape}")
            tmp = self.ai_layers[idx](x_list[idx])
            # print(f"shape tmp={tmp.shape}")
            y_list += [tmp]
        y_list = mx.array(y_list).T
        y_list = self.out_proj(y_list)
        # print(f"shape y_list={y_list.shape}")
        y_pattern = mx.concatenate(y_list)
        # print(f"shape y_pattern={y_pattern.shape}")
        return y_pattern
