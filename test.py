# -*- coding: utf-8 -*-
# @Time : 2022/11/11 21:26
# @Author : KKKc
# @FileName: test.py
import torch
import torch.nn as nn

bce = nn.CrossEntropyLoss()

pre = torch.Tensor([[0.2,0.1,0.3,0.4]])
target = torch.Tensor([1])
bce(pre,target)