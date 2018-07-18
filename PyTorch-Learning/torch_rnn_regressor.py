#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-9 下午4:40
@email: lph0729@163.com  

"""
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.001


def plot_finc():
    y_sin = np.sin(np_data)
    y_cos = np.cos(np_data)

    plt.plot(np_data, y_sin, "r-", label="sin")
    plt.plot(np_data, y_cos, "r-", label="cos")
    plt.legend(loc="best")
    plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        """
        x shape: (batch, time_step, input_size)
        h_state：(n_layers, batch, hidden_size)
        rnn_out: (batch, time_step, hidden_size)
        """
        rnn_out, h_state = self.rnn(x, h_state)

        outs = []
        for time_step in range(rnn_out.size(1)):
            outs.append(self.out(rnn_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


if __name__ == '__main__':
    np_data = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    h_state = None

    plt.figure(1, figsize=(12, 5))
    plt.ion()

    for step in range(100):
        start, end = step * np.pi, (step + 1) * np.pi
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        x_sin = np.sin(steps)
        y_cos = np.cos(steps)

        x_sin_pro = torch.from_numpy(x_sin[np.newaxis, :, np.newaxis])
        y_cos_pro = torch.from_numpy(y_cos[np.newaxis, :, np.newaxis])

        prediction, h_state = rnn(Variable(x_sin_pro), h_state)
        h_state = Variable(h_state.data)

        loss = loss_func(prediction, Variable(y_cos_pro))
        print("loss:", loss.data.numpy()[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.plot(steps, y_cos, "r-")
        plt.plot(steps, prediction.data.numpy().flatten(), "b-")
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
