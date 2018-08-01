#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-30 下午2:34
@email: lph0729@163.com  

"""
from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import torch

N_SAMPLES = 20
N_HIDDEN = 300


def main():
    # training data
    x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), dim=1)
    y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
    x = Variable(x)
    y = Variable(y)

    # test data
    test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), dim=1)
    test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
    test_x = Variable(test_x)
    test_y = Variable(test_y)

    # show data
    # plt.scatter(x.data.numpy(), y.data.numpy(), c="magenta", s=50, alpha=0.5, label="train")
    # plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c="cyan", s=50, alpha=0.5, label="test")
    # plt.legend(loc="upper left")
    # plt.ylim(-2.5, 2.5)
    # plt.show()

    net_overfitting = nn.Sequential(
        nn.Linear(1, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1)
    )

    net_dropped = nn.Sequential(
        nn.Linear(1, N_HIDDEN),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1)
    )

    print("net_overfitting:\n", net_overfitting, "\nnet_dropped:\n", net_dropped)

    optimizer_overfit = optim.Adam(net_overfitting.parameters(), lr=0.01)
    optimizer_drop = optim.Adam(net_dropped.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    plt.ion()

    for t in range(500):
        pred_overfit = net_overfitting(x)
        pred_drop = net_dropped(x)

        loss_overfit = loss_func(pred_overfit, y)
        loss_drop = loss_func(pred_drop, y)

        optimizer_overfit.zero_grad()
        optimizer_drop.zero_grad()
        loss_overfit.backward()
        loss_drop.backward()
        optimizer_overfit.step()
        optimizer_drop.step()

        if t % 10 == 0:
            net_overfitting.eval()
            net_dropped.eval()

            plt.cla()
            test_pred_overfit = net_overfitting(test_x)
            test_pred_drop = net_dropped(test_x)

            plt.scatter(x.data.numpy(), y.data.numpy(), c="magenta", s=50, alpha=0.5, label="train")
            plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c="cyan", s=50, alpha=0.5, label="test")
            plt.plot(test_x.data.numpy(), test_pred_overfit.data.numpy(), "r--", lw=3, label="overfitting")
            plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), "b--", lw=3, label="dropout(50%)")

            plt.text(0, -1.2, "overfitting loss=%.4f" % loss_func(test_pred_overfit, test_y).data.numpy(),
                     fontdict={"size": 20, "color": "red"})
            plt.text(0, -1.5, "dropout loss=%.4f" % loss_func(test_pred_drop, test_y).data.numpy(),
                     fontdict={"size": 20, "color": "blue"})

            plt.legend(loc="upper left")

            net_overfitting.train()
            net_dropped.train()

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
