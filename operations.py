import torch
import torch.nn as nn
import math

iDim = 90

OPS = {
    'base_layer': lambda C_in, C_out, data_shape, kernel_size, ac_func, pooling, bn: Baselayer(C_in, C_out,
                                                                                               data_shape,
                                                                                               kernel_size,
                                                                                               ac_func,
                                                                                               pooling,
                                                                                               bn,
                                                                                               stride=1,
                                                                                               padding=0),
    'fc2': lambda C_in, C_out, ac_func: FC2(C_in, C_out, ac_func),
    'fc3': lambda C_in, C_out, ac_func: FC3(C_in, C_out, ac_func),
    'fc4': lambda C_in, C_out, ac_func: FC4(C_in, C_out, ac_func),
    'fc5': lambda C_in, C_out, ac_func: FC5(C_in, C_out, ac_func),
    'identity': lambda C_in, C_out, kernel_size: Identity(C_in, C_out, kernel_size, 1, 0)
}

k_s = [math.floor(iDim / 8), math.floor(2 * iDim / 8), math.floor(3 * iDim / 8), math.floor(4 * iDim / 8),
       math.floor(5 * iDim / 8), math.floor(6 * iDim / 8), math.floor(7 * iDim / 8), iDim]


class Baselayer(nn.Module):

    def __init__(self, C_in, C_out, data_shape, kernel_size, ac_func, pooling, bn, stride, padding):
        super(Baselayer, self).__init__()
        self.c1 = C_in
        self.c2 = C_out
        self.c3 = kernel_size
        self.c4 = ac_func
        self.c6 = pooling
        self.c7 = bn
        self.cc = nn.ModuleList()
        self.cc.append(
            nn.Conv2d(C_in, C_out, kernel_size=(1, k_s[kernel_size]), stride=stride, padding=padding, bias=False))
        if bn == 1:
            self.cc.append(
                nn.BatchNorm2d(C_out, affine=True))
        if ac_func == 1:
            self.cc.append(nn.ReLU())
        if ac_func == 2:
            self.cc.append(nn.LeakyReLU(0.2))
        if ac_func == 3:
            self.cc.append(nn.LeakyReLU(0.33))
        if ac_func == 4:
            self.cc.append(nn.Tanh())
        if pooling == 1:
            self.cc.append(
                nn.MaxPool2d(kernel_size=(1, data_shape - k_s[kernel_size] + 1), stride=stride, padding=padding))
        if pooling == 2:
            self.cc.append(
                nn.AvgPool2d(kernel_size=(1, data_shape - k_s[kernel_size] + 1), stride=stride, padding=padding))

    def forward(self, x):
        for i in range(len(self.cc)):
            name = self.cc[i].__class__.__name__
            if name == 'BatchNorm2d' and x.shape[0] == 1:
                print("error, the number of batch is 1!")
            else:
                x = self.cc[i](x)
        return x



class FC2(nn.Module):
    def __init__(self, C_in, C_out, ac_func):
        super(FC2, self).__init__()
        self.cc = nn.ModuleList()
        self.cc.append(nn.Linear(C_in, 64))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(64, C_out))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.cc)):
            x = self.cc[i](x)
        return x


class FC3(nn.Module):
    def __init__(self, C_in, C_out, ac_func):
        super(FC3, self).__init__()
        self.cc = nn.ModuleList()
        self.cc.append(nn.Linear(C_in, 64))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(64, 32))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(32, C_out))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.cc)):
            x = self.cc[i](x)
        return x


class FC4(nn.Module):
    def __init__(self, C_in, C_out, ac_func):
        super(FC4, self).__init__()
        self.cc = nn.ModuleList()
        self.cc.append(nn.Linear(C_in, 64))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(64, 32))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(32, 16))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(16, C_out))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.cc)):
            x = self.cc[i](x)
        return x


class FC5(nn.Module):
    def __init__(self, C_in, C_out, ac_func):
        super(FC5, self).__init__()
        self.cc = nn.ModuleList()
        self.cc.append(nn.Linear(C_in, 64))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(64, 32))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(32, 16))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(16, 8))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())
        self.cc.append(nn.Linear(8, C_out))
        if ac_func == 0:
            self.cc.append(nn.ReLU())
        if ac_func == 1:
            self.cc.append(nn.LeakyReLU(0.3))
        if ac_func == 2:
            self.cc.append(nn.Tanh())
        if ac_func == 3:
            self.cc.append(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.cc)):
            x = self.cc[i](x)
        return x


class Identity(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Identity, self).__init__()
        self.cc = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.cc(x)
        return x
