#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-27 下午3:15
@email: lph0729@163.com  

"""
import torch
from torchvision import datasets, transforms
from torch.utils import data
from torch import nn
from torch.autograd import Variable

EPOCH = 1
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = False


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(50 * 7 * 7, 10)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)


def main():
    train_data = datasets.MNIST(
        root="./model_datas/MnistDatasets",
        train=True,
        transform=transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_data = datasets.MNIST(
        root="./model_datas/MnistDatasets",
        train=False
    )

    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255.0
    test_y = test_data.test_labels[:2000].cuda()

    cnn = CNN()
    cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            batch_x, batch_y = Variable(x.cuda()), Variable(y.cuda())

            y_pred = cnn(batch_x)
            loss = loss_func(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backend()
            optimizer.step()

            if step % 50 == 0:
                test_y_pred = cnn(Variable(test_x))
                test_y_pred = torch.max(test_y_pred, 1)[1].cuda().data.squeeze()
                accuracy = torch.sum(test_y_pred == test_y) / test_y.size(0)
                print("Epoch:\n", epoch, "\ntrain loss:%0.4f\n" % loss.data.numpy(),
                      "\ntest_accuracy:0.2f\n" % accuracy.data.numpy())


if __name__ == '__main__':
    main()
