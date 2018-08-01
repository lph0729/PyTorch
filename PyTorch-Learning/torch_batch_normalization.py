#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-31 上午10:54
@email: lph0729@163.com  

"""
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.utils import data
from torch.optim import Adam
from torch.autograd import Variable
import torch
import numpy as np
from matplotlib import pyplot as plt

N_SAMPLES = 2000
N_HIDDEN = 8
BATCH_SIZE = 64
EPOCH = 12
ACTIVATION = F.tanh
B_INIT = -0.2
LR = 0.03


class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)
        self.fcs = []
        self.bns = []

        for i in range(N_HIDDEN):  # build hidden layers and BN layer
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)
            setattr(self, "fc%i" % i, fc)  # important set layer to the module
            self._set_init(fc)
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, "bn%i" % i, bn)
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)
        self._set_init(self.predict)

    def _set_init(self, layer):
        init.normal(layer.weight, mean=0., std=.1)
        init.constant(layer.bias, B_INIT)

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)  # input batch normalization
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            pre_activation.append(x)
            if self.do_bn: x = self.bns[i](x)  # batch normalization
            x = ACTIVATION(x)
            layer_input.append(x)
        out = self.predict(x)

        return out, layer_input, pre_activation


def generate_datas():
    # training data
    x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
    noise = np.random.normal(0, 2, x.shape)
    y = np.square(x) - 5 + noise

    # test data
    test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
    noise = np.random.normal(0, 2, test_x.shape)
    test_y = np.square(test_x) - 5 + noise

    train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    test_x, test_y = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()
    test_data = [test_x, test_y]

    train_dataset = data.TensorDataset(train_x, train_y)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # show data
    plt.scatter(x, y, c="#FF9359", s=50, alpha=0.2, label="train")
    plt.legend(loc="upper left")
    # plt.savefig("./model_datas/pictures/batch_normal_trian_data.png")
    # plt.show()

    return train_loader, test_data


def plot_histogram(axs, l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10)
            the_range = (-7, 10)
        else:
            p_range = (-4, 4)
            the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(())
            a.set_xticks(())
        ax_pa_bn.set_xticks(p_range)
        ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel('PreAct')
        axs[1, 0].set_ylabel('BN PreAct')
        axs[2, 0].set_ylabel('Act')
        axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)


def build_nn_model(train_loader, test_data, nets, loss_func, optimizers, axs):
    losses = [[], []]  # 记录两个神经网络的损失值
    for epoch in range(EPOCH):
        layer_inputs, pre_acts = [], []
        for net, loss in zip(nets, losses):
            net.eval()  # eval主要是去除移动的均值和变量
            pred, layer_input, pre_act = net(Variable(test_data[0]))

            loss.append(loss_func(pred, Variable(test_data[1])))
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)

            net.train()
        plot_histogram(axs, *layer_inputs, *pre_acts)

        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = Variable(b_x), Variable(b_y)
            for net, opt in zip(nets, optimizers):
                pred, _, _, = net(b_x)
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    plt.figure(2)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.ylim((0, 2000))
    plt.legend(loc='best')


def main():
    # 1.构建训练与测试数据
    train_loader, test_data = generate_datas()

    # 2.构建神经网络
    nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
    print(*nets)

    # 3.定义目标函数以及优化器
    optimizers = [Adam(net.parameters(), lr=LR) for net in nets]
    loss_func = nn.MSELoss()

    # 4.展示神经网络
    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()  # something about plotting
    plt.show()

    # 5.训练模型以及展示模型
    build_nn_model(train_loader, test_data, nets, loss_func, optimizers, axs)


if __name__ == '__main__':
    main()
