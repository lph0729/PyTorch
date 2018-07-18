#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-4 下午10:00
@email: lph0729@163.com  

"""
from torch.utils import data
from torchvision import datasets, transforms
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss
from torch.optim import Adam
from torch.autograd import Variable
from matplotlib import cm, pyplot as plt
from sklearn.manifold import TSNE
import torch
import os

LR = 0.01
BATCH_SIZE = 32
DOWNLOAD_MINIST = False
EPOCHS = 1
HAS_SK = True

if not os.listdir("./model_datas/MnistDatasets/"):
    DOWNLOAD_MINIST = True


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            ReLU(),
            MaxPool2d(kernel_size=2)
        )
        self.conv_2 = Sequential(
            Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            ReLU(),
            MaxPool2d(kernel_size=2)
        )
        self.out = Linear(32 * 7 * 7, 10)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        fcl_input = conv_2.view(conv_2.size(0), -1)
        fcl_output = self.out(fcl_input)
        return fcl_output, fcl_input


def plot_with_lables(lowDWeights, labels):
    plt.cla()
    x, y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(x, y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("visualize last layer")
    plt.show()
    plt.pause(0.01)


if __name__ == '__main__':
    # 1.Mnist digits dataset
    train_data = datasets.MNIST(
        root="./model_datas/MnistDatasets/",
        train=True,  # true: training data  False: testing data
        transform=transforms.ToTensor(),
        # Convert a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
        # and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MINIST
    )
    # print(train_data.train_data.type)  # type: torch.ByteTensor
    # plot one example
    # print("train_data_featrues:", train_data.train_data.size(),
    #       "\ntrain_data_labels:", train_data.train_labels.size())

    # plt.imshow(train_data.train_data[0].numpy(), cmap="gray")
    # plt.title("%i" % train_data.train_labels[0])
    # plt.show()

    # the image batch shape will be (50, 1, 28, 28)
    data_loader = data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_data = datasets.MNIST(
        root="./model_datas/MnistDatasets/",
        train=False,
        # transform=transforms.ToTensor()
    )
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255.
    test_y = test_data.test_labels

    # 2.build cnn
    cnn = CNN()
    # print(cnn)
    optimizer = Adam(cnn.parameters(), lr=LR)
    loss_func = CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for step, (b_x, b_y) in enumerate(data_loader):
            batch_x, batch_y = Variable(b_x), Variable(b_y)
            pred_y = cnn(batch_x)[0]
            loss = loss_func(pred_y, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output, last_layer = cnn(Variable(test_x))
                test_y_pred = torch.max(test_output, 1)[1].data.squeeze().numpy()
                accuracy = float((test_y_pred == test_y.numpy()).astype(int).sum()) / float(test_y.size(0))
                print("epoch:", epoch, "|step:", step, "|train_loss:", loss.data.numpy(),
                      "|test_acuracy:%.2f" % accuracy)

                if HAS_SK:
                    # Visualization of trainned flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plot_with_lables(low_dim_embs, labels)

    plt.ioff()

    # print 10 predictions from test data
    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
    print(pred_y, "predict_number")
    print(test_y[:10].numpy(), "real number")
