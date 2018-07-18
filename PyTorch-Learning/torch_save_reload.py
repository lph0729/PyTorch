#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-4 上午10:51
@email: lph0729@163.com  

"""
from torch.nn import Sequential, Linear, ReLU, MSELoss
from torch.autograd import Variable
from torch.optim import SGD
import torch
from matplotlib import pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), 1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)


def save_model():
    net_1 = Sequential(
        Linear(1, 10),
        ReLU(),
        Linear(10, 1)
    )

    optimizer = SGD(net_1.parameters(),lr=0.5)
    loss_func = MSELoss()

    for step in range(500):
        y_pred = net_1(x)
        loss = loss_func(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("net_1")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), y_pred.data.numpy(), "r-", lw=5)

    # save the model using two methods
    torch.save(net_1, "./model_datas/models/net_1.pkl")  # save entire net

    torch.save(net_1.state_dict(), "./model_datas/models/net_1_params.pkl")  # save only the params of the model


def restore_net():
    # restore entire net_1 to net_2
    net_2 = torch.load("./model_datas/models/net_1.pkl")
    y_pred = net_2(x)

    # plot result
    plt.subplot(132)
    plt.title("net_2")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), y_pred.data.numpy(), "r-", lw=5)


def restore_params():
    # restore net_1's params in net_1 to net_3
    net_3 = Sequential(
        Linear(1, 10),
        ReLU(),
        Linear(10, 1)
    )

    net_3.load_state_dict(torch.load("./model_datas/models/net_1_params.pkl"))
    y_pred = net_3(x)

    plt.subplot(133)
    plt.title("net_3")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), y_pred.data.numpy(), "r-", lw=5)

    plt.savefig("./model_datas/pictures/torch_save_restore.png")
    plt.show()


if __name__ == '__main__':
    save_model()
    restore_net()
    restore_params()