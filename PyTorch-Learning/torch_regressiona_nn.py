#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-2 上午10:27
@email: lph0729@163.com  

"""
from torch.autograd import Variable
from torch.nn import Module, Linear, MSELoss
from torch.nn import functional as F
from torch.optim import SGD
from matplotlib import pyplot as plt
import torch


class Net(Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = Linear(n_feature, n_hidden)
        self.predict = Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(200,1)

    """Torch can only train on Variable, so convert them to Variable.
        The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors."""
    x, y = Variable(x), Variable(y)

    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
    print(net)  # net structure 这里就体现出来动态建图了，你还可以传入其他的参数来改变网络的结构

    optimizer = SGD(net.parameters(), lr=0.5)  # net.parameters(): 记录每个网络层参数的状态
    loss_fun = MSELoss()

    plt.ion()  # show the process of the figure

    for step in range(500):
        y_pred = net.forward(x)
        loss = loss_fun(y_pred, y)
        optimizer.zero_grad()  # clear gradients for next train 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # Does the update

        if step % 5 == 0:
            plt.cla()  # Clear axis
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), y_pred.data.numpy(), "r-", lw=3)
            plt.text(0.5, 0, "loss=%.4f" % loss.data.numpy(), fontdict={"size": 20, "color": "red"})
            plt.pause(0.1)

    plt.ioff()

    plt.savefig("./model_datas/regressor.png")
    plt.show()
    plt.close()
