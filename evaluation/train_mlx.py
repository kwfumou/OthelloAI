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
    
    def __call__(self,x):
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
        super(OthelloAI,self).__init__()
        self.ai_layers: List[OthelloAILayer] = []
        for in_dims in in_dims_list:
            self.ai_layers.append(OthelloAILayer(in_dims=in_dims,out_dims=1))
        self.out_proj = nn.Linear(in_dims, out_dims)
    
    def __call__(self,x_list):

        y_list = []
        for layers in self.ai_layers:
            y_list.append(layers(x_list))
        # Concatenateに該当する関数を探そう
        y_pattern = Concatenate(axis=-1)(y_list)
        return self.out_proj(y_pattern)

"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from argparse import ArgumentParser
from functools import partial

class OthelloAILayer(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
        super(OthelloAILayer, self).__init__()

        self.in_proj = nn.Linear(in_dims, hidden_dims)
        self.active = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_proj = nn.Linear(hidden_dims, hidden_dims)
        self.out_proj = nn.Linear(hidden_dims, out_dims)
    
    def forward(self, x):
        x = self.in_proj(x)
        x = self.active(x)
        x = self.hidden_proj(x)
        x = self.active(x)
        x = self.out_proj(x)
        x = self.active(x)
        return x

class OthelloAI(nn.Module):
    def __init__(self, in_dims1: int, in_dims2: int, out_dims: int):
        super(OthelloAI, self).__init__()
        self.ai_layers1 = OthelloAILayer(in_dims1, out_dims)
        self.ai_layers2 = OthelloAILayer(in_dims2, out_dims)
        self.combined_proj = nn.Linear(out_dims * 2, out_dims)
    
    def forward(self, x1, x2):
        x1 = self.ai_layers1(x1)
        x2 = self.ai_layers2(x2)
        x_combined = mx.concat([x1, x2], axis=-1)
        return self.combined_proj(x_combined)

# Usage example
model = OthelloAI(in_dims1=10, in_dims2=20, out_dims=5)
input1 = mx.random.normal((batch_size, 10))
input2 = mx.random.normal((batch_size, 20))
output = model(input1, input2)

"""

def main():
    print("main")

    layer_sizes = [1,2,3,4,5]
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        print(f"in_dim={in_dim}")
        print(f"out_dim={out_dim}")

    # model = MyMLP(
    #     x_dim=,
    #     h_dim=16,
    #     out_dim=1,
    # )
    # mx.eval(model.parameters())

    # optimizer = optim.Adam(learning_rate=0.001)

    # state = [model.state, optimizer.state, mx.random.state]

    # @partial(mx.compile, inputs=state, outputs=state)
    # def step():
    #     loss_and_grad_fn = nn.value_and_grad(gcn, forward_fn)
    #     (loss, y_hat), grads = loss_and_grad_fn(
    #         gcn, x, adj, y, train_mask, args.weight_decay
    #     )
    #     optimizer.update(gcn, grads)
    #     return loss, y_hat

if __name__ == "__main__":

    main()
