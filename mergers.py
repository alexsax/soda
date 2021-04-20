import torch
import torch.nn as nn
from typing import Optional, Mapping, Union, Tuple

from .base import Merger


# class Merger:
#     def __init__(self, n: int = 1):
#         super().__init__()
#         self.preds = []
#         self.n = n
# 
#     def append(self, x):
#         self.preds.append(x)
# 
#     @property
#     def result(self):
#         pass

class MeanMerger(Merger):

    @classmethod
    # From: https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
    def nanmean(cls, v, *args, inplace=False, **kwargs):
        if not inplace:
            v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        # return v.mean(*args, **kwargs)
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    @property
    def result(self):
        stacked = torch.stack(self.preds, dim=1)
        pred = MeanMerger.nanmean(stacked, dim=1, keepdim=False)
        # print(stacked.min(), '->', pred.min())
        return pred



class MedianMerger(Merger):
    @property
    def result(self):
        stacked = torch.stack(self.preds, dim=1)
        pred, _ = torch.nanmedian(stacked, dim=1, keepdim=False)
        # print(stacked.min(), '->', pred.min())
        return pred

