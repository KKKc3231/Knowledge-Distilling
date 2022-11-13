# -*- coding: utf-8 -*-
# @Time : 2022/11/12 10:36
# @Author : KKKc
# @FileName: train_kd.py


from tqdm import tqdm
import torch.nn as nn
from config import *
from model import Student,Teacher
from Dataset import Get_loader
import torch.nn.functional as F

# 获取数据集
train_dataloader,test_dataloader = Get_loader()

# 初始化网络
model_teacher = Teacher()
model_student = Student()

# 加载模型参数
model_teacher.load_state_dict(torch.load("./save_model/Techer_31_0.9846.pth",map_location=DEVICE))
model_student.load_state_dict(torch.load("./save_model/kd_Student_31_0.9197.pth",map_location=DEVICE))
model_student.to(device=DEVICE)
model_teacher.to(device=DEVICE)
model_teacher.eval()

# 参数
T = 5
alpha = 0.1

# 损失函数
hard_loss = nn.CrossEntropyLoss()
soft_loss = nn.KLDivLoss(reduction="batchmean")

# 优化函数
optimal = torch.optim.Adam(model_student.parameters(),lr=LEARNING_RATE)

for epoch in range(EPOCH):
    model_student.train()
    for data,target in tqdm(train_dataloader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        # Teacher prediction
        with torch.no_grad():
            pre_t = model_teacher(data)

        # Student prediction
        pre_s = model_student(data)
        h_loss = hard_loss(pre_s,target)
        #
        s_loss = soft_loss(
            F.softmax(pre_s/T,dim=1),
            F.softmax(pre_t/T,dim=1)
        )

        loss = alpha * h_loss + (1 - alpha) * s_loss
        optimal.zero_grad()
        loss.backward()
        optimal.step()

    # test
    model_student.eval()
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for x,y in test_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pre = model_student(x)
            predictions = pre.max(1).indices
            num_correct += (predictions==y).sum()
            # print(num_correct)
            num_sample += predictions.size(0)
            # print(num_sample)
        acc  = (num_correct / num_sample).item()

    model_student.train()
    print("Epoch:{}----Accuracy:{:4f}".format(epoch + 1, acc))
    if epoch % 10 == 0:
        torch.save(model_student.state_dict(),"./save_model/kd_Student_{}_{}.pth".format(epoch + 1,acc))