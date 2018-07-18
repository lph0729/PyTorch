#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-1 下午6:53
@email: lph0729@163.com  

"""
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt

# generate datas
train_data = torch.linspace(-5, 5, 200)

torch_var = Variable(train_data)
x_np_data = torch_var.data.numpy()

# following are popular activation functions
y_relu = F.relu(torch_var).data.numpy()
y_sigmoid = F.sigmoid(torch_var).data.numpy()
y_tanh = F.tanh(torch_var).data.numpy()
y_softplus = F.softplus(torch_var).data.numpy()

# plt to visualize these activation functions
plt.figure(1, figsize=(8, 6))
# plotting relu figure
plt.subplot(221)
plt.plot(x_np_data, y_relu, c="red", label="relu")
plt.ylim((-1, 5))
plt.legend(loc="best")

# plotting sigmoid figure
plt.subplot(222)
plt.plot(x_np_data, y_sigmoid, c="red", label="sigmoid")
plt.ylim((-0.2, 1.2))
plt.legend(loc="best")

# plotting tanh figure
plt.subplot(223)
plt.plot(x_np_data, y_tanh, c="red", label="tanh")
plt.ylim((-1.2, 1.2))
plt.legend(loc="best")

# plotting relu figure
plt.subplot(224)
plt.plot(x_np_data, y_softplus, c="red", label="softplus")
plt.ylim((-0.2, 6))
plt.legend(loc="best")

plt.savefig("./model_datas/activates.png")
plt.show()
plt.close()
