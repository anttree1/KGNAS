import torch
import torch.nn as nn
from operations import *
import random
import utils as ut


def E2E(input):
    return torch.add(input, input.permute(0, 1, 3, 2))


def E2N(input):
    return input.permute(0, 1, 3, 2) * 2


class NetworkESS(nn.Module):
    def __init__(self, list_arch, data_shape, drop_out):
        super(NetworkESS, self).__init__()
        # in, out, data_shape, kernel_size, ac_func, pooling, bn
        self.layers_E1 = OPS['base_layer'](1, list_arch[0][4], data_shape, list_arch[0][0] - 1, list_arch[0][1],
                                           list_arch[0][2], list_arch[0][3])
        self.layers_E2 = OPS['base_layer'](list_arch[0][4], list_arch[1][4], data_shape, list_arch[1][0] - 1,
                                           list_arch[1][1], list_arch[1][2], list_arch[1][3])
        self.layers_E3 = OPS['base_layer'](list_arch[1][4], list_arch[2][4], data_shape, list_arch[2][0] - 1,
                                           list_arch[2][1], list_arch[2][2], list_arch[2][3])
        self.layers_E1_3 = OPS['base_layer'](list_arch[0][4], list_arch[3][4], data_shape, list_arch[3][0] - 1,
                                             list_arch[3][1], list_arch[3][2], list_arch[3][3])
        self.layers_N1 = OPS['base_layer'](list_arch[2][4] + list_arch[3][4], list_arch[4][4], data_shape,
                                           list_arch[4][0] - 1,
                                           list_arch[4][1], list_arch[4][2], list_arch[4][3])
        self.layers_N2_4 = OPS['base_layer'](list_arch[1][4], list_arch[5][4], data_shape,
                                             list_arch[5][0] - 1,
                                             list_arch[5][1], list_arch[5][2], list_arch[5][3])
        self.layers_G1 = OPS['base_layer'](list_arch[4][4] + list_arch[5][4], list_arch[6][4], data_shape,
                                           list_arch[6][0] - 1,
                                           list_arch[6][1], list_arch[6][2], list_arch[6][3])
        if list_arch[7][0] == 2:
            self.layer_FC = OPS['fc2'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 3:
            self.layer_FC = OPS['fc3'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 4:
            self.layer_FC = OPS['fc4'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 5:
            self.layer_FC = OPS['fc5'](list_arch[6][4], 2, list_arch[7][1])
        self.dropout1 = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1)
        self.out_size = list_arch[6][4]

    def forward(self, x):
        layer1 = E2E(self.layers_E1(x))
        layer1 = self.dropout1(layer1)
        layer2 = E2E(self.layers_E2(layer1))
        layer2 = self.dropout1(layer2)
        layer3 = E2E(self.layers_E3(layer2))
        layer3 = self.dropout1(layer3)
        layer4 = E2E(self.layers_E1_3(layer1))
        layer4 = self.dropout1(layer4)
        input_5 = torch.cat([layer3, layer4], 1)
        layer5 = E2N(self.layers_N1(input_5))
        layer5 = self.dropout1(layer5)
        layer6 = E2N(self.layers_N2_4(layer2))
        layer6 = self.dropout1(layer6)
        input_7 = torch.cat([layer5, layer6], 1)
        layer7 = self.layers_G1(input_7)
        layer7 = self.dropout1(layer7)
        layer8 = self.layer_FC(layer7.view(-1, self.out_size))
        layer8 = self.dropout1(layer8)
        out = self.softmax(layer8)
        return out


class NetworkSW(nn.Module):
    def __init__(self, list_arch, data_shape, drop_out):
        super(NetworkSW, self).__init__()
        # in, out, data_shape, kernel_size, ac_func, pooling, bn
        self.layers_E1 = OPS['base_layer'](1, list_arch[0][4], data_shape, list_arch[0][0] - 1, list_arch[0][1],
                                           list_arch[0][2], list_arch[0][3])
        self.layers_E2 = OPS['base_layer'](list_arch[0][4], list_arch[1][4], data_shape, list_arch[1][0] - 1,
                                           list_arch[1][1], list_arch[1][2], list_arch[1][3])
        self.layers_E3 = OPS['base_layer'](list_arch[1][4], list_arch[2][4], data_shape, list_arch[2][0] - 1,
                                           list_arch[2][1], list_arch[2][2], list_arch[2][3])
        self.layers_E1_3 = OPS['base_layer'](list_arch[0][4], list_arch[3][4], data_shape, list_arch[3][0] - 1,
                                             list_arch[3][1], list_arch[3][2], list_arch[3][3])
        self.layers_N1 = OPS['base_layer'](list_arch[2][4] + list_arch[3][4], list_arch[4][4], data_shape,
                                           list_arch[4][0] - 1,
                                           list_arch[4][1], list_arch[4][2], list_arch[4][3])
        self.layers_N2_4 = OPS['base_layer'](list_arch[1][4], list_arch[5][4], data_shape,
                                             list_arch[5][0] - 1,
                                             list_arch[5][1], list_arch[5][2], list_arch[5][3])
        self.layers_G1 = OPS['base_layer'](list_arch[4][4] + list_arch[5][4], list_arch[6][4], data_shape,
                                           list_arch[6][0] - 1,
                                           list_arch[6][1], list_arch[6][2], list_arch[6][3])
        if list_arch[7][0] == 2:
            self.layer_FC = OPS['fc2'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 3:
            self.layer_FC = OPS['fc3'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 4:
            self.layer_FC = OPS['fc4'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 5:
            self.layer_FC = OPS['fc5'](list_arch[6][4], 2, list_arch[7][1])
        self.dropout1 = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1)
        self.out_size = list_arch[6][4]

    def forward(self, x):
        layer1 = E2E(self.layers_E1(x))
        layer1 = self.dropout1(layer1)
        layer2 = E2E(self.layers_E2(layer1))
        layer2 = self.dropout1(layer2)
        layer3 = E2E(self.layers_E3(layer2))
        layer3 = self.dropout1(layer3)
        layer4 = E2E(self.layers_E1_3(layer1))
        layer4 = self.dropout1(layer4)
        input_5 = torch.cat([layer3, layer4], 1)
        layer5 = E2N(self.layers_N1(input_5))
        layer5 = self.dropout1(layer5)
        layer6 = E2N(self.layers_N2_4(layer2))
        layer6 = self.dropout1(layer6)
        input_7 = torch.cat([layer5, layer6], 1)
        layer7 = self.layers_G1(input_7)
        layer7 = self.dropout1(layer7)
        layer8 = self.layer_FC(layer7.view(-1, self.out_size))
        layer8 = self.dropout1(layer8)
        out = self.softmax(layer8)
        return out, layer4


class NetworkKDE(nn.Module):
    def __init__(self, list_arch, data_shape, drop_out):
        super(NetworkKDE, self).__init__()
        # in, out, data_shape, kernel_size, ac_func, pooling, bn
        self.layers_E1 = OPS['base_layer'](1, list_arch[0][4], data_shape, list_arch[0][0] - 1, list_arch[0][1],
                                           list_arch[0][2], list_arch[0][3])
        self.layers_E2 = OPS['base_layer'](list_arch[0][4], list_arch[1][4], data_shape, list_arch[1][0] - 1,
                                           list_arch[1][1], list_arch[1][2], list_arch[1][3])
        self.layers_E3 = OPS['base_layer'](list_arch[1][4], list_arch[2][4], data_shape, list_arch[2][0] - 1,
                                           list_arch[2][1], list_arch[2][2], list_arch[2][3])
        self.layers_E1_3 = OPS['base_layer'](list_arch[0][4], list_arch[3][4], data_shape, list_arch[3][0] - 1,
                                             list_arch[3][1], list_arch[3][2], list_arch[3][3])
        self.layers_N1 = OPS['base_layer'](list_arch[2][4] + list_arch[3][4], list_arch[4][4], data_shape,
                                           list_arch[4][0] - 1,
                                           list_arch[4][1], list_arch[4][2], list_arch[4][3])
        self.layers_N2_4 = OPS['base_layer'](list_arch[1][4], list_arch[5][4], data_shape,
                                             list_arch[5][0] - 1,
                                             list_arch[5][1], list_arch[5][2], list_arch[5][3])
        self.layers_G1 = OPS['base_layer'](list_arch[4][4] + list_arch[5][4], list_arch[6][4], data_shape,
                                           list_arch[6][0] - 1,
                                           list_arch[6][1], list_arch[6][2], list_arch[6][3])
        if list_arch[7][0] == 2:
            self.layer_FC = OPS['fc2'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 3:
            self.layer_FC = OPS['fc3'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 4:
            self.layer_FC = OPS['fc4'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 5:
            self.layer_FC = OPS['fc5'](list_arch[6][4], 2, list_arch[7][1])
        self.dropout1 = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1)
        self.out_size = list_arch[6][4]

    def forward(self, x):
        layer1 = E2E(self.layers_E1(x))
        layer1 = self.dropout1(layer1)
        layer2 = E2E(self.layers_E2(layer1))
        layer2 = self.dropout1(layer2)
        layer3 = E2E(self.layers_E3(layer2))
        layer3 = self.dropout1(layer3)
        layer4 = E2E(self.layers_E1_3(layer1))
        layer4 = self.dropout1(layer4)
        input_5 = torch.cat([layer3, layer4], 1)
        layer5 = E2N(self.layers_N1(input_5))
        layer5 = self.dropout1(layer5)
        layer6 = E2N(self.layers_N2_4(layer2))
        layer6 = self.dropout1(layer6)
        input_7 = torch.cat([layer5, layer6], 1)
        layer7 = self.layers_G1(input_7)
        layer7 = self.dropout1(layer7)
        layer8 = self.layer_FC(layer7.view(-1, self.out_size))
        layer8 = self.dropout1(layer8)
        out = self.softmax(layer8)
        return out, layer1, layer2, layer3


class NetworkSC(nn.Module):
    def __init__(self, list_arch, data_shape, drop_out):
        super(NetworkSC, self).__init__()
        # in, out, data_shape, kernel_size, ac_func, pooling, bn
        self.layers_E1 = OPS['base_layer'](1, list_arch[0][4], data_shape, list_arch[0][0] - 1, list_arch[0][1],
                                           list_arch[0][2], list_arch[0][3])
        self.layers_E2 = OPS['base_layer'](list_arch[0][4], list_arch[1][4], data_shape, list_arch[1][0] - 1,
                                           list_arch[1][1], list_arch[1][2], list_arch[1][3])
        self.layers_E3 = OPS['base_layer'](list_arch[1][4], list_arch[2][4], data_shape, list_arch[2][0] - 1,
                                           list_arch[2][1], list_arch[2][2], list_arch[2][3])
        self.layers_E1_3 = OPS['base_layer'](list_arch[0][4], list_arch[3][4], data_shape, list_arch[3][0] - 1,
                                             list_arch[3][1], list_arch[3][2], list_arch[3][3])
        self.layers_N1 = OPS['base_layer'](list_arch[2][4] + list_arch[3][4], list_arch[4][4], data_shape,
                                           list_arch[4][0] - 1,
                                           list_arch[4][1], list_arch[4][2], list_arch[4][3])
        self.layers_N2_4 = OPS['base_layer'](list_arch[1][4], list_arch[5][4], data_shape,
                                             list_arch[5][0] - 1,
                                             list_arch[5][1], list_arch[5][2], list_arch[5][3])
        self.layers_G1 = OPS['base_layer'](list_arch[4][4] + list_arch[5][4], list_arch[6][4], data_shape,
                                           list_arch[6][0] - 1,
                                           list_arch[6][1], list_arch[6][2], list_arch[6][3])
        if list_arch[7][0] == 2:
            self.layer_FC = OPS['fc2'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 3:
            self.layer_FC = OPS['fc3'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 4:
            self.layer_FC = OPS['fc4'](list_arch[6][4], 2, list_arch[7][1])
        if list_arch[7][0] == 5:
            self.layer_FC = OPS['fc5'](list_arch[6][4], 2, list_arch[7][1])
        self.dropout1 = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1)
        self.out_size = list_arch[6][4]

    def forward(self, x):
        layer1 = E2E(self.layers_E1(x))
        layer1 = self.dropout1(layer1)
        layer2 = E2E(self.layers_E2(layer1))
        layer2 = self.dropout1(layer2)
        layer3 = E2E(self.layers_E3(layer2))
        layer3 = self.dropout1(layer3)
        layer4 = E2E(self.layers_E1_3(layer1))
        layer4 = self.dropout1(layer4)
        input_5 = torch.cat([layer3, layer4], 1)
        layer5 = E2N(self.layers_N1(input_5))
        layer5 = self.dropout1(layer5)
        layer6 = E2N(self.layers_N2_4(layer2))
        layer6 = self.dropout1(layer6)
        input_7 = torch.cat([layer5, layer6], 1)
        layer7 = self.layers_G1(input_7)
        layer7 = self.dropout1(layer7)
        layer8 = self.layer_FC(layer7.view(-1, self.out_size))
        layer8 = self.dropout1(layer8)
        out = self.softmax(layer8)
        return out, layer3, input_5