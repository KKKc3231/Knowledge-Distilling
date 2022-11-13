# -*- coding: utf-8 -*-
# @Time : 2022/11/11 20:49
# @Author : KKKc
# @FileName: train.py

import torch
from tqdm import tqdm
import torch.nn as nn
from config import *
# from torchinfo import summary
from model import Student,Teacher
from Dataset import Get_loader
import torch.nn.functional as F

# 获取数据集
train_dataloader,test_dataloader = Get_loader()
# 初始化网络
model_techer = Teacher()
model_student = Student()
model_techer.to(device=DEVICE)
model_student.to(device=DEVICE)

# summary(model_techer)
# summary(model_student)

# 损失函数
hard_loss = nn.CrossEntropyLoss()

# 优化函数
optimal = torch.optim.Adam(model_techer.parameters(),lr=LEARNING_RATE)

# 先训练Teacher
for epoch in range(EPOCH):
    model_techer.train()
    for data,target in tqdm(train_dataloader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        pre = model_techer(data)
        loss = hard_loss(pre,target) # 一个batch更新一次参数
        # print(loss)
        # 反向传播，优化权重
        optimal.zero_grad()
        loss.backward()
        optimal.step()

    model_techer.eval()
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for x,y in test_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pre = model_techer(x)
            predictions = pre.max(1).indices
            num_correct += (predictions==y).sum()
            # print(num_correct)
            num_sample += predictions.size(0)
            # print(num_sample)
        acc  = (num_correct / num_sample).item()
    model_techer.train()
    print("Epoch:{}----Accuracy:{}".format(epoch+1,acc))
    if epoch % 10 == 0:
        torch.save(model_techer.state_dict(),"./save_model/Techer_{}_{}.pth".format(epoch+1,acc))


