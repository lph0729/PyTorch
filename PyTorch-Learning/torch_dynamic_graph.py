#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-26 下午5:14
@email: lph0729@163.com

"""
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
from matplotlib import pyplot as plt

INPUT_SIZE = 1
HIDDEN_SIZE = 32
LR = 0.02


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, h_state):
        """
        x shape: (batch, time_step, input_size)
        h_state：(n_layers, batch, hidden_size)
        r_out: (batch, time_step, hidden_size)
        """
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))

        print(torch.stack(outs, dim=1).size())
        return torch.stack(outs, dim=1), h_state


def main():
    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    h_state = None

    plt.figure(1, figsize=(12, 5))
    plt.ion()  # continuously plot

    step = 0
    for i in range(100):
        # 下边这三行主要是实现动态神经网络
        dynamic_steps = np.random.randint(1, 4)
        start, end = step * np.pi, (step + dynamic_steps) * np.pi
        step += dynamic_steps

        steps = np.linspace(start, end, 10 * dynamic_steps, dtype=np.float32)
        print("训练数据的time_step:", steps)

        x_np = np.sin(steps)
        y_np = np.cos(steps)

        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

        y_pred, h_state = rnn(Variable(x), h_state)
        h_state = Variable(h_state.data)

        loss = loss_func(y_pred, Variable(y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.plot(steps, y_np.flatten(), "r-")
        plt.plot(steps, y_pred.data.numpy().flatten(), "b-")
        plt.draw()
        plt.pause(0.1)

    plt.ioff()

    plt.savefig("./model_datas/pictures/dynamic_graph.png")
    plt.show()


if __name__ == '__main__':
    main()
