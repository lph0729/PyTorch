#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-4 下午4:13
@email: lph0729@163.com  

"""
from torch.autograd import Variable
from torch.utils import data
from torch.nn import Module, Linear, MSELoss
from torch.nn import functional as F
from torch.optim import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import torch

LR = 0.01
BATCH_SIZE = 30
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

torch_dataset = data.TensorDataset(x, y)
loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = Linear(1, 20)
        self.out_layer = Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        y_pred = self.out_layer(x)
        return y_pred


if __name__ == '__main__':
    # different nets
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()

    # different optimizer
    optimizer_SGD = SGD(net_SGD.parameters(), lr=LR)
    optimizer_Momentum = SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    optimizer_RMSprop = RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    optimizer_Adam = Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

    loss_func = MSELoss()

    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
    optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_RMSprop, optimizer_Adam]
    loss_his = [[], [], [], []]

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            x_var, y_var = Variable(x), Variable(y)
            for net, opt, loss_per in zip(nets, optimizers, loss_his):
                predict = net(x_var)
                loss = loss_func(predict, y_var)

                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_per.append(loss.data.numpy())

    labels = ["SGD", "Momentum", "RMSprop", "Adam"]
    for i, loss in enumerate(loss_his):
        plt.plot(loss, label=labels[i])

    plt.legend(loc="best")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.ylim(0, 0.2)
    plt.savefig("./model_datas/pictures/torch_optimizer.png")
    plt.show()


