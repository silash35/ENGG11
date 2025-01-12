import time

import torch


def dydx(x, y):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def mean_square(x):
    return torch.mean(x**2)


def timer(func, *args):
    start_time = time.monotonic()
    result = func(*args)
    elapsed_time = time.monotonic() - start_time
    return result, elapsed_time
