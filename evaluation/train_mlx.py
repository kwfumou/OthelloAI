import mlx.core as mx
import mlx.nn as nn

class MyMLP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
        super().__init__()

        self.in_proj = nn.Linear(in_dims, hidden_dims)
        self.out_proj = nn.Linear(hidden_dims, out_dims)

    def __call__(self, x):
        x = self.in_proj(x)
        x = mx.maximum(x, 0)
        return self.out_proj(x)

model = MyMLP(2, 1)

# All the model parameters are created but since MLX is lazy by
# default, they are not evaluated yet. Calling `mx.eval` actually
# allocates memory and initializes the parameters.
mx.eval(model.parameters())

# Setting a parameter to a new value is as simply as accessing that
# parameter and assigning a new array to it.
model.in_proj.weight = model.in_proj.weight * 2
mx.eval(model.parameters())