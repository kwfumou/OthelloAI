import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from argparse import ArgumentParser
from functools import partial
from typing import List


class OthelloAILayer(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
        super(OthelloAILayer, self).__init__()

        self.in_proj = nn.Linear(in_dims, hidden_dims)
        self.active = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_proj = nn.Linear(hidden_dims, hidden_dims)
        self.out_proj = nn.Linear(hidden_dims, out_dims)

    def __call__(self, x):
        # xの方が何かを調べる？？
        # それはあとでいいんじゃね？
        x = self.in_proj(x)
        x = self.active(x)
        x = self.hidden_proj(x)
        x = self.active(x)
        x = self.out_proj(x)
        x = self.active(x)
        return x


class OthelloAI(nn.Module):
    def __init__(self, in_dims_list: List[int], out_dims: int, hidden_dims: int):
        super(OthelloAI, self).__init__()
        self.ai_layers: List[OthelloAILayer] = []
        for in_dims in in_dims_list:
            self.ai_layers.append(OthelloAILayer(in_dims=in_dims, out_dims=1))
        self.out_proj = nn.Linear(in_dims, out_dims)

    def __call__(self, x_list):

        y_list = []
        for layers in self.ai_layers:
            y_list.append(layers(x_list))
        # Concatenateに該当する関数を探そう
        y_pattern = mx.concatenate(
            arrays=y_list,
            axis=-1,
        )
        return self.out_proj(y_pattern)
