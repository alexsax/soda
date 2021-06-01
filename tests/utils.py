import torch

def equal_except_for_nan(x, y):
    x = x.detach().clone()
    y = y.detach().clone()
    isnan = (x != x)
    isnan[y != y] = 1.0
    return torch.allclose(x[~isnan], y[~isnan])