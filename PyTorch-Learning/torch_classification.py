#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-2 下午5:47
@email: lph0729@163.com  

"""
from torch.nn import functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.optim import SGD
import torch


class Net(Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = Linear(n_feature, n_hidden)
        self.pred = Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.pred(x)
        return x


if __name__ == '__main__':
    n_data = torch.ones(100, 2)

    x_0 = torch.normal(2 * n_data, 1)  # first param: means  second:std
    y_0 = torch.zeros(100)

    x_1 = torch.normal(-2 * n_data, 1)
    y_1 = torch.ones(100)

    x = torch.cat((x_0, x_1), 0).type(torch.FloatTensor)  # first:input second:dim=0
    y = torch.cat((y_0, y_1), ).type(torch.FloatTensor)
    print(x[98:102])

    x, y = Variable(x), Variable(y)

    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap="RdYlGn")
    # plt.savefig("./model_datas/classification.png")
    # plt.show()

    net = Net(n_feature=2, n_hidden=10, n_output=2)

    optimizer = SGD(net.parameters(), lr=0.02)
    loss_func = CrossEntropyLoss()

    plt.ion()

    for step in range(500):
        y_pred = net.forward(x)

        loss = loss_func(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            plt.cla()
            pred = torch.max(y_pred, 1)[1]
            pred_y = pred.data.numpy().sequeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap="RdYlGn")
            accuracy = float((pred_y==target_y).astype(int).sum())/float(target_y.size)
            plt.text(1.5, -4, "Accuracy=%.2f" % accuracy, fontdict={"size":20, "color": "red"})
            plt.pause(0.1)

    plt.ioff()
    plt.savefig("classification_model.png")
    plt.show()
