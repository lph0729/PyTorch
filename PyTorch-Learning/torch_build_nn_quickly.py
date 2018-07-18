#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-3 下午6:06
@email: lph0729@163.com  

"""
from torch.nn import Linear, Module, Sequential, ReLU
from torch.nn import functional as F


# method1: build nn
class Net(Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = Linear(n_feature, n_hidden)
        self.pred = Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.pred(x)
        return x


net_1 = Net(n_feature=2, n_hidden=10, n_output=2)
print(net_1)

# method2: build nn
net_2 = Sequential(
    Linear(1, 10),
    ReLU(),
    Linear(10, 1)
)

print(net_2)

"""net_1 architecture:

Net (
  (hidden): Linear (2 -> 10)
  (pred): Linear (10 -> 2)
)
"""


"""net_2 architecture:

Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""