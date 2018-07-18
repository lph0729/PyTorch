#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-1 上午11:56
@email: lph0729@163.com  

"""

import torch
import numpy as np

"""主要知识点：numpy与torch数据格式之间的转化以及各种运算
    1.from_numpy(): 将numpy的ndarray数据格式转化为torch的LongTensor数据格式
    2.numpy(): 将torch的LongTensor数据格式转化为numpy的ndarray数据格式
"""
np_data_1 = np.arange(6).reshape(2, 3)
np_data_2 = np.array([-2, -5, 2, 4])
list_data_1 = [-2, -5, 2, 4]
list_data_2 = [[1, 2], [3, 4]]

# 1.numpy与torch数据格式之间的转化
np_to_torch_data_1 = torch.from_numpy(np_data_1)

torch_to_np_data_1 = np_to_torch_data_1.numpy()

# 2.torch的基本运算
np_to_torch_data_2 = torch.from_numpy(np_data_2)

# abs
torch_abs = np_to_torch_data_2.abs()

torch_float_data = torch.FloatTensor(list_data_1)  # 数据的类型有int转化为float类型

# sin
np_sin_data = np.sin(np_data_2)
torch_sin_data = torch.sin(torch_float_data)

# mean
np_mean_data = np.mean(np_data_2)
torch_mean_data = torch.mean(torch_float_data)

# matrix multiplication
torch_data = torch.FloatTensor(list_data_2)
np_mm_data = np.matmul(list_data_2, list_data_2)
torch_mm_data = torch.mm(torch_data, torch_data)

# dot
np_dot_data = np.dot(list_data_2, list_data_2)
torch_dot_data = torch.dot(torch_data, torch_data) # 会将乘积的每一项求和输出一维数据

print("np_data", np_data_1,
      "\nnp_to_torch_data:", np_to_torch_data_1,
      "\ntorch_to_np_data:", torch_to_np_data_1,
      "\ntorch_abs:", torch_abs,
      "\nnp_sin_data:", np_sin_data,
      "\ntorch_sin_data:", torch_sin_data,
      "\nnp_mean_data:", np_mean_data,
      "\ntorch_mean_data:", torch_mean_data,
      "\nnp_mm_data:", np_mm_data,
      "\ntorch_mm_data:", torch_mm_data,
      "\nnp_dot_data:", np_dot_data,
      "\ntorch_dot_data:", torch_dot_data,
      )
