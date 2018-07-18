#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-1 下午5:42
@email: lph0729@163.com  

"""
from torch.autograd import Variable
import torch

""" Variable in torch is to build a computational graph,
    but this graph is dynamic compared with a static graph in Tensorlow or Theano.
    So torch does not have placeholder, torch can just pass variable to the computational graph."""

torch_tensor = torch.FloatTensor([[1, 2], [3, 4]])  # build tensor
torch_var = Variable(torch_tensor, requires_grad=True)  # build a variable , usually for computer gradients

"""till now the tensor and variable seem the same.
    however, the variable is a part of the graph, it's a part of the auto-gradient"""
torch_tensor_out = torch.mean(torch_tensor * torch_tensor)
torch_var_out = torch.mean(torch_var * torch_var)

torch_var_out.backward()  # backpropagation from torch_var_out

torch_grad = torch_var.grad  # he gradient w.r.t variable, d(torch_var_out)/d(torch_var) = 1/4*2*torch_var = 1/2*torch
torch_var_data = torch_var.data  # this is data in torch tensor format
torch_var_data_to_numpy = torch_var_data.numpy()  # this is data in ndarray format

print("torch_tensor:", torch_tensor,
      "\ntorch_var:", torch_var,
      "\ntorch_out:", torch_tensor_out,
      "\ntorch_var_out:", torch_var_out,
      "\ntorch_grad:", torch_grad,
      "\ntorch_var_data:", torch_var_data,
      "\ntorch_var_data_to_numpy:", torch_var_data_to_numpy)
