from copy import deepcopy

# import tensorflow as tf
# from tensorflow.keras.layers import Add, Dense, Input, LeakyReLU, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tqdm import trange, tqdm
from random import shuffle
import subprocess
import datetime
from copy import deepcopy
from othello_py import *
import mlx.core as mx
import mlx.nn as nn

from othelloAI import OthelloAI
import const
import mlx.optimizers as optim
from functools import partial
from mlx.utils import tree_flatten
import time


def digit(n, r):
    n = str(n)
    l = len(n)
    for _ in range(r - l):
        n = "0" + n
    return n


# 棋譜から盤面データを作る
records = []
# for num in range(1):
#     with open("self_play/" + digit(num, 7) + ".txt", "r") as f:
#         records.extend(list(f.read().splitlines()))
# with open("self_play/0.txt", "r") as f:
with open("self_play/0000000.txt", "r") as f:
    records.extend(list(f.read().splitlines()))
records = [records[idx] for idx in range(int(len(records) / 2))]
data = []
evaluate_additional = subprocess.Popen(
    "./evaluate.out".split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE
)
for record in tqdm(records):
    o = othello()
    board_data = []
    for i in range(0, len(record), 2):
        board_str = ""
        for j in range(hw):
            for k in range(hw):
                board_str += (
                    "."
                    if o.grid[j][k] == vacant or o.grid[j][k] == legal
                    else str(o.grid[j][k])
                )
        evaluate_additional.stdin.write(
            (str(o.player) + "\n" + board_str + "\n").encode("utf-8")
        )
        evaluate_additional.stdin.flush()
        additional_param = [
            float(elem)
            for elem in evaluate_additional.stdout.readline().decode().split()
        ]
        board_data.append(
            [
                board_str,
                o.player,
                additional_param[0],
                additional_param[1],
                additional_param[2],
            ]
        )
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        x = ord(record[i]) - ord("a")
        y = int(record[i + 1]) - 1
        o.move(y, x)
    result = o.n_stones[0] - o.n_stones[1]
    n_vacant = hw2 - (o.n_stones[0] + o.n_stones[1])
    if result > 0:
        result += n_vacant
    elif result < 0:
        result -= n_vacant
    for board_datum in board_data:
        board_datum.append(result)
        data.append(board_datum)
evaluate_additional.kill()
print("n_data", len(data))
breakpoint()
# 学習
test_ratio = 0.1
n_epochs = 10

diagonal8_idx = [[0, 9, 18, 27, 36, 45, 54, 63], [7, 14, 21, 28, 35, 42, 49, 56]]
for pattern in deepcopy(diagonal8_idx):
    diagonal8_idx.append(list(reversed(pattern)))

edge_2x_idx = [
    [9, 0, 1, 2, 3, 4, 5, 6, 7, 14],
    [9, 0, 8, 16, 24, 32, 40, 48, 56, 49],
    [49, 56, 57, 58, 59, 60, 61, 62, 63, 54],
    [54, 63, 55, 47, 39, 31, 23, 15, 7, 14],
]
for pattern in deepcopy(edge_2x_idx):
    edge_2x_idx.append(list(reversed(pattern)))

triangle_idx = [
    [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],
    [0, 8, 16, 24, 1, 9, 17, 2, 10, 3],
    [7, 6, 5, 4, 15, 14, 13, 23, 22, 31],
    [7, 15, 23, 31, 6, 14, 22, 5, 13, 4],
    [63, 62, 61, 60, 55, 54, 53, 47, 46, 39],
    [63, 55, 47, 39, 62, 54, 46, 61, 53, 60],
    [56, 57, 58, 59, 48, 49, 50, 40, 41, 32],
    [56, 48, 40, 32, 57, 49, 41, 58, 50, 59],
]

pattern_idx = [diagonal8_idx, edge_2x_idx, triangle_idx]
ln_in = sum([len(elem) for elem in pattern_idx]) + 1
all_data = [[] for _ in range(ln_in)]
all_labels = []


def make_lines(board, patterns, player):
    res = []
    for pattern in patterns:
        tmp = []
        for elem in pattern:
            tmp.append(1.0 if board[elem] == str(player) else 0.0)
        for elem in pattern:
            tmp.append(1.0 if board[elem] == str(1 - player) else 0.0)
        res.append(tmp)
    return res


def calc_n_stones(board):
    res = 0
    for elem in board:
        res += int(elem != ".")
    return res


def collect_data(board, player, v1, v2, v3, result):
    global all_data, all_labels
    v1 = float(v1)
    v2 = float(v2)
    v3 = float(v3)
    result = float(result) / 64
    player = int(player)
    idx = 0
    for i in range(len(pattern_idx)):
        lines = make_lines(board, pattern_idx[i], 0)
        for line in lines:
            all_data[idx].append(line)
            idx += 1
    all_data[idx].append([v1 / 30, (v2 - 15) / 15, (v3 - 15) / 15])
    all_labels.append(result)


x = [None for _ in range(ln_in)]
ys = []
names = ["diagonal8", "edge2X", "triangle"]
idx = 0

for i in trange(len(data)):
    collect_data(*data[i])
len_data = len(all_labels)
print(len_data)

tmp_data = deepcopy(all_data)
tmp_labels = deepcopy(all_labels)
all_data = [[] for _ in range(len(tmp_data))]
all_labels = []
shuffled = list(range(len_data))
shuffle(shuffled)
for i in shuffled:
    all_labels.append(tmp_labels[i])
    for j in range(len(tmp_data)):
        all_data[j].append(tmp_data[j][i])

all_data = [np.array(arr) for arr in all_data]
all_labels = np.array(all_labels)

n_train_data = int(len_data * (1.0 - test_ratio))
n_test_data = int(len_data * test_ratio)

# train_data = [arr[0:n_train_data] for arr in all_data]
train_data = [mx.array(arr[0:n_train_data]) for arr in all_data]
train_labels = mx.array(all_labels[0:n_train_data])
# fixed_train_data = [
#     [mx.array(train_data[tp_idx][idx]) for tp_idx in range(len(train_data))]
#     for idx in range(len(train_labels))
# ]
# train_data = fixed_train_data
test_data = [mx.array(arr[n_train_data:len_data]) for arr in all_data]
test_labels = mx.array(all_labels[n_train_data:len_data])
# fixed_test_data = [
#     [mx.array(test_data[tp_idx][idx]) for tp_idx in range(len(test_data))]
#     for idx in range(len(test_labels))
# ]
# test_data = fixed_test_data
print(f"len train={len(train_data)}")
# breakpoint()


def loss_fn(y_hat, y, parameters=None):
    # return mx.mean(nn.losses.cross_entropy(y_hat, y))
    print(f" y={y}")
    print(f"shape y={y.shape}")
    print(f"shape y_hat={y_hat.shape}")
    return (y_hat - y).square().mean()


def forward_fn(model, x, label):
    # y_hat = model(graph.edge_index, graph.node_features, graph.batch_indices)
    y_hat = model(x)
    loss = loss_fn(y_hat, label, model.parameters())
    return loss, y_hat


device = mx.gpu  # or mx.cpu
mx.set_default_device(device)

mx.random.seed(42)
in_dims_list = [all_data[pt_idx].shape[1] for pt_idx in range(const.pattern_size)]
oai = OthelloAI(in_dims_list=in_dims_list, out_dims=1, hidden_dims=16)
mx.eval(oai.parameters())
optimizer = optim.Adam(learning_rate=const.lr)
loss_and_grad_fn = nn.value_and_grad(oai, forward_fn)


def train(train_loader, train_labels):
    loss_sum = 0.0
    # for idx, dt in enumerate(train_loader):

    (loss, y_hat), grads = loss_and_grad_fn(
        model=oai,
        x=train_loader,
        label=train_labels,
    )
    optimizer.update(oai, grads)
    mx.eval(oai.parameters(), optimizer.state)
    loss_sum += loss.item()
    return loss_sum / len(train_loader)


def test(test_loader, test_label):
    loss_sum = 0.0
    # for dt in test_loader:
    (loss, y_hat), grads = loss_and_grad_fn(
        model=oai,
        x=test_loader,
        label=test_label,
    )
    loss_sum += loss.item()
    return loss_sum / len(test_loader)


def epoch():

    loss = train(train_data, train_labels)
    test_loss = test(test_data, test_labels)
    return loss, test_loss


epochs = 30
best_test_loss = 1000.0
for e in range(epochs):
    start_time = time.time()
    loss, test_loss = epoch()
    best_test_loss = min(best_test_loss, test_loss)
    end_time = time.time()

    print(
        " | ".join(
            [
                f"Epoch: {e:3d}",
                f"Train loss: {loss:.3f}",
                f"Test loss: {test_loss:.3f}",
                f"time={end_time-start_time:.3f}",
            ]
        )
    )
# print(f"\n==> Best test accuracy: {best_test_acc:.3f}")
