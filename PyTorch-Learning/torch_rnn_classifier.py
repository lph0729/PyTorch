#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-6 上午10:18
@email: lph0729@163.com  

"""
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils import data
from torch.optim import Adam
import torch.nn as nn
import torch

EPOCHS = 1
DOWNLOAD_MNIST = False
BATCH_SIZE = 50
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        """r_out: shape (batch, time_step, output_size)
            h_n: shape (n_layers, batch, hidden_size)
            c_n: shape (n_layers, batch, hidden_size)"""
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        rnn_out = self.out(r_out[:, -1, :])
        return rnn_out


if __name__ == '__main__':
    # 1.load mnist datastes
    train_data = datasets.MNIST(
        root="./model_datas/MnistDatasets/",
        train=True,
        transform=transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )

    data_loader = data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_data = datasets.MNIST(
        root="./model_datas/MnistDatasets/",
        train=False,
        # transform=transforms.ToTensor()
    )

    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255.
    test_y = test_data.test_labels

    # 2.build the neural network
    rnn = RNN()
    print(rnn)

    optimizer = Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for step, (batch_x, batch_y) in enumerate(data_loader):
            b_x, b_y = Variable(batch_x), Variable(batch_y)
            squ_b_x = torch.squeeze(b_x, dim=1)
            predict = rnn(squ_b_x)

            loss = loss_func(predict, b_y)
            print("loss:", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_x = Variable(torch.squeeze(test_x, dim=1))
                # test_x = torch.squeeze(test_x, dim=1)
                pred_output = rnn(test_x)
                pred_y = torch.max(pred_output, 1)[1].data.squeeze().numpy()
                accuracy = float((test_y.numpy() == pred_y).astype(int).sum()) / float(test_y.size(0))
                print("epoch:", epoch, "|step:", step, "|train_loss:", loss.data.numpy(),
                      "|test_acuracy:%.2f" % accuracy)

    # print 10 predictions from test data
    test_output, _ = rnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
    print(pred_y, "predict_number")
    print(test_y[:10].numpy(), "real number")
