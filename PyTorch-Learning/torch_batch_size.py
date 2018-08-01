#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-4 下午3:10
@email: lph0729@163.com  

"""
from torch.utils import data
import torch

BATCH_SIZE = 2

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = data.TensorDataset(data_tensor=x, target_tensor=y)
loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

for epoch in range(3):  # entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):
        print(
            "epoch:", epoch, "|step:", step, "|batch_x:", batch_x.numpy(), "|batch_y:", batch_y.numpy()
        )


