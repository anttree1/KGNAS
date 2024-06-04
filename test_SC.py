import torch
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import sys
import torch.nn as nn
import scipy.io as sio
import model_search as ms
import utils as ut
import matplotlib.pyplot as plt
from torch.autograd import Variable

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--fold', type=int, default=5, help='Number of Cross-validation')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--learning_rate', type=float, default=3e-2, help='learning rage')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=51, help='epoch')
parser.add_argument('--drop_size', type=int, default=0.1, help='normal_shape')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--s_id', type=int, default=125, help='sick id')
parser.add_argument('--h_id', type=int, default=206, help='health id')
parser.add_argument('--normal_shape', type=int, default=90, help='normal_shape')
parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_aal_10/ALLASD{}_aal.mat')
parser.add_argument('--save_add_input', type=str,
                    default='/home/ai/data/wangxingy/work4/data/test_sw/input_matrix/input_matrix{}.edge')
args = parser.parse_args()


def main_arch(list_arch):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    epoch_list = []
    load_data = sio.loadmat(args.data_address.format(args.fold))
    X_train = get_batch(
        torch.tensor(load_data['net_train']).view(-1, 1, args.normal_shape, args.normal_shape).type(
            torch.FloatTensor))
    Y_train = get_batch(load_data['phenotype_train'][:, 2].astype(float))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = ms.NetworkSC(list_arch, args.normal_shape, drop_out=args.drop_size)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        epoch_list.append(epoch)
        train_loss, train_acc, c1_layer, c0_layer = train(X_train, Y_train, model, optimizer, criterion)
        if epoch == 50:
            de_matrix_non = c1_layer[args.s_id][0] - c0_layer[args.h_id][0]
            de_matrix_sc = c1_layer[args.s_id][1] - c0_layer[args.h_id][1]
            de_matrix_non = torch.sum(de_matrix_non, dim=1).squeeze().detach()
            de_matrix_sc = torch.sum(de_matrix_sc, dim=1).squeeze().detach()
            de_matrix_non = de_matrix_non.cpu()
            de_matrix_sc = de_matrix_sc.cpu()
            de_matrix_sc = np.array(de_matrix_sc)
            de_matrix_non = np.array(de_matrix_non)
            de_matrix_sc = get_diag(de_matrix_sc)
            de_matrix_non = get_diag(de_matrix_non)
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(de_matrix_non, cmap='viridis', interpolation='nearest')
            plt.title('Heatmap of Tensor non')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(de_matrix_sc, cmap='viridis', interpolation='nearest')
            plt.title('Heatmap of Tensor sc')
            plt.colorbar()
            plt.show()
            sc = get_max(de_matrix_sc, 0.015)
            non = get_max(de_matrix_non, 0.015)
            write_matrix_to_file(sc, 'sc.txt')
            write_matrix_to_file(non, 'non.txt')


def train(X_train, Y_train, model, optimizer, criterion):
    loss_AM = ut.AvgrageMeter()
    acc_AM = ut.AvgrageMeter()
    list_out_1 = []
    list_out_2 = []
    for steps, (input, target) in enumerate(zip(X_train, Y_train)):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=True).cuda()
        target = Variable(target, requires_grad=True).cuda()
        optimizer.zero_grad()
        logits, non_layer, sc_layer = model(input)
        loss = criterion(logits, target.long())
        loss.backward()
        optimizer.step()
        accuracy = ut.accuracy(logits, target)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
        if target == 1.:
            list_out_1.append([non_layer, sc_layer])
        if target == 0.:
            list_out_2.append([non_layer, sc_layer])
    return loss_AM.avg, acc_AM.avg, list_out_1, list_out_2


def get_batch(X_input):
    X_output = torch.utils.data.DataLoader(X_input, batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                           num_workers=0)
    return X_output


def get_max(matrix, pro):
    num_elements_to_keep = int(np.prod(matrix.shape) * pro)
    sorted_elements = np.sort(matrix.flatten())[::-1]
    threshold = sorted_elements[num_elements_to_keep - 1]
    result_matrix = np.where(matrix >= threshold, matrix, 0)
    return result_matrix


def get_diag(matrix):
    matrix_min, matrix_max = matrix.min(), matrix.max()
    matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
    np.fill_diagonal(matrix, 1)
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            avg_value = (matrix[i, j] + matrix[j, i]) / 2
            matrix[i, j] = avg_value
            matrix[j, i] = avg_value
    return matrix


def write_matrix_to_file(matrix, file_path):
    with open(file_path, 'w') as f:
        for row in matrix:
            row_str = ' '.join(map(str, row))
            f.write(row_str + '\n')


def boundary(matrix, b):
    matrix[matrix < b] = 0
    return matrix


main_arch([[8, 1, 1, 1, 8], [4, 2, 2, 1, 32], [3, 2, 1, 2, 4], [6, 4, 2, 1, 16], [1, 3, 2, 2, 16], [7, 2, 2, 2, 32],
           [6, 4, 2, 1, 8], [2, 1]])
