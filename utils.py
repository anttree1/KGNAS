import random
from material import *
import numpy as np
from sklearn.metrics import roc_auc_score



def individual_list(individual_try):
    list_out = []
    E1_kernel = E2E_type_list[int(individual_try[:7], 2)][1]  # 卷积核类型编号
    E1_ac = E2E_type_list[int(individual_try[:7], 2)][2]  # 激活函数编号
    E1_pool = E2E_type_list[int(individual_try[:7], 2)][3]  # 池化编号
    E1_bn = E2E_type_list[int(individual_try[:7], 2)][4]  # 批归一化
    E1_num = number_list[int(individual_try[7:10], 2)]  # 具体通道数
    list_out.append([E1_kernel, E1_ac, E1_pool, E1_bn, E1_num])

    E2_kernel = E2E_type_list[int(individual_try[10:17], 2)][1]
    E2_ac = E2E_type_list[int(individual_try[10:17], 2)][2]
    E2_pool = E2E_type_list[int(individual_try[10:17], 2)][3]
    E2_bn = E2E_type_list[int(individual_try[10:17], 2)][4]
    E2_num = number_list[int(individual_try[17:20], 2)]
    list_out.append([E2_kernel, E2_ac, E2_pool, E2_bn, E2_num])

    E3_kernel = E2E_type_list[int(individual_try[20:27], 2)][1]
    E3_ac = E2E_type_list[int(individual_try[20:27], 2)][2]
    E3_pool = E2E_type_list[int(individual_try[20:27], 2)][3]
    E3_bn = E2E_type_list[int(individual_try[20:27], 2)][4]
    E3_num = number_list[int(individual_try[27:30], 2)]
    list_out.append([E3_kernel, E3_ac, E3_pool, E3_bn, E3_num])

    E1_3_kernel = E2E_type_list[int(individual_try[30:37], 2)][1]
    E1_3_ac = E2E_type_list[int(individual_try[30:37], 2)][2]
    E1_3_pool = E2E_type_list[int(individual_try[30:37], 2)][3]
    E1_3_bn = E2E_type_list[int(individual_try[30:37], 2)][4]
    E1_3_num = number_list[int(individual_try[37:40], 2)]
    list_out.append([E1_3_kernel, E1_3_ac, E1_3_pool, E1_3_bn, E1_3_num])

    N1_kernel = E2N_type_list[int(individual_try[40:47], 2)][1]
    N1_ac = E2N_type_list[int(individual_try[40:47], 2)][2]
    N1_pool = E2N_type_list[int(individual_try[40:47], 2)][3]
    N1_bn = E2N_type_list[int(individual_try[40:47], 2)][4]
    N1_num = number_list[int(individual_try[47:50], 2)]
    list_out.append([N1_kernel, N1_ac, N1_pool, N1_bn, N1_num])

    N2_4_kernel = E2N_type_list[int(individual_try[50:57], 2)][1]
    N2_4_ac = E2N_type_list[int(individual_try[50:57], 2)][2]
    N2_4_pool = E2N_type_list[int(individual_try[50:57], 2)][3]
    N2_4_bn = E2N_type_list[int(individual_try[50:57], 2)][4]
    N2_4_num = number_list[int(individual_try[57:60], 2)]
    list_out.append([N2_4_kernel, N2_4_ac, N2_4_pool, N2_4_bn, N2_4_num])

    G1_kernel = N2G_type_list[int(individual_try[60:67], 2)][1]
    G1_ac = N2G_type_list[int(individual_try[60:67], 2)][2]
    G1_pool = N2G_type_list[int(individual_try[60:67], 2)][3]
    G1_bn = N2G_type_list[int(individual_try[60:67], 2)][4]
    G1_num = number_list[int(individual_try[67:70], 2)]
    list_out.append([G1_kernel, G1_ac, G1_pool, G1_bn, G1_num])

    FC_layer = FC_type_list[int(individual_try[70:], 2)][1]
    FC_ac = FC_type_list[int(individual_try[70:], 2)][2]
    list_out.append([FC_layer, FC_ac])
    return list_out


def accuracy(output, target):
    maxk = max((1,))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].contiguous().view(-1).float().sum(0)
    correct_k.mul_(100.0 / batch_size)
    return correct_k


def sensitivity(output, target):
    maxk = max((1,))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    true_positives = correct.mul(target.view(1, -1))
    sensitivity = true_positives.sum().float() / target.sum().float()
    return sensitivity


def specificity(output, target):
    maxk = max((1,))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    true_negatives = correct.mul(1 - target.view(1, -1))
    specificity = true_negatives.sum().float() / (batch_size - target.sum().float())
    return specificity


def AUC(output, target):
    maxk = max((1,))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    output = output.cpu()
    output = output.detach().numpy()
    pred = pred.cpu()
    pred = pred.detach().numpy().reshape(-1)
    answer = []
    for var_p, var_out in zip(pred, output):
        if var_p == 0:
            answer.append(1 - var_out[0])
        else:
            answer.append(var_out[1])
    target = target.cpu()
    tr = target.detach().numpy().reshape(-1)
    ar = np.array(answer)
    auc = roc_auc_score(tr, ar)
    return auc


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
