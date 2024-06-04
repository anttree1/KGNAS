import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import sys
import torch.nn as nn
import scipy.io as sio
import model_search as ms
import utils as ut
from torch.autograd import Variable

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--fold', type=int, default=5, help='Number of Cross-validation')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rage')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=201, help='epoch')
parser.add_argument('--drop_size', type=int, default=0, help='normal_shape')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--normal_shape', type=int, default=90, help='normal_shape')
parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_aal_10/ALLASD{}_aal.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_cc200_10/ALLASD{}_cc200.mat')
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
    X_test = get_batch(
        torch.tensor(load_data['net_test']).view(-1, 1, args.normal_shape, args.normal_shape).type(
            torch.FloatTensor))
    Y_train = get_batch(load_data['phenotype_train'][:, 2].astype(float))
    Y_test = get_batch(load_data['phenotype_test'][:, 2].astype(float))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = ms.NetworkKDE(list_arch, args.normal_shape, drop_out=args.drop_size)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        epoch_list.append(epoch)
        train_loss, train_acc, layer1, layer2, layer3 = train(X_train, Y_train, model, optimizer, criterion)
        test_loss, test_acc = infer(X_test, Y_test, model, criterion)
        if epoch == 20:
            layer3 = layer3.flatten().detach().cpu().numpy()
            sns.kdeplot(layer3)
        if epoch == 200:
            layer3 = layer3.flatten().detach().cpu().numpy()
            sns.kdeplot(layer3)
            plt.show()
        print('epoch', epoch)
        print('train_acc', train_acc)
        print('test_acc', test_acc)


def train(X_train, Y_train, model, optimizer, criterion):
    loss_AM = ut.AvgrageMeter()
    acc_AM = ut.AvgrageMeter()
    for steps, (input, target) in enumerate(zip(X_train, Y_train)):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=True).cuda()
        target = Variable(target, requires_grad=True).cuda()
        optimizer.zero_grad()
        logits, layer1, layer2, layer3 = model(input)
        loss = criterion(logits, target.long())
        loss.backward()
        optimizer.step()
        accuracy = ut.accuracy(logits, target)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
    return loss_AM.avg, acc_AM.avg, layer1, layer2, layer3


def infer(X_vaild, Y_vaild, model, criterion):
    loss_AM = ut.AvgrageMeter()
    acc_AM = ut.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(zip(X_vaild, Y_vaild)):
        with torch.no_grad():
            input = Variable(input).cuda()
        with torch.no_grad():
            target = Variable(target).cuda()

        logits, layer1, layer2, layer3 = model(input)
        loss = criterion(logits, target.long())

        accuracy = ut.accuracy(logits, target)
        n = input.size(0)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
        '''
        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, loss_AM.avg, acc_AM.avg)
        '''

    return loss_AM.avg, acc_AM.avg


def get_batch(X_input):
    X_output = torch.utils.data.DataLoader(X_input, batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                           num_workers=0)
    return X_output


def get_matrix_max(input_matrix, va):
    flattened_matrix = input_matrix.flatten()

    values, indices = torch.topk(flattened_matrix, va)
    rows = indices // 90  # 行坐标
    cols = indices % 90  # 列坐标
    top_coordinates = list(zip(rows.tolist(), cols.tolist()))
    out_coo = {tuple(sorted(t)) for t in top_coordinates}
    filtered_tuples = filter(lambda x: x[0] != x[1], out_coo)
    merged_tuples = list(filtered_tuples)
    return merged_tuples


def get_edge_matrix(list1, matrix_size, i):
    matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    for x, y in list1:
        matrix[y - 1][x - 1] = 1
        matrix[x - 1][y - 1] = 1
    with open(args.save_add_input.format(i), 'w') as file:
        for row in matrix:
            file.write(' '.join(str(num) for num in row) + '\n')


def get_matrix_top(input_matrix, va):
    top_10_coords_list = []
    for i in range(input_matrix.size(0)):
        values, indices = torch.topk(input_matrix[i], va)
        row_coords = [(i, int(index)) for index in indices]
        top_10_coords_list.extend(row_coords)
    out_coo = {tuple(sorted(t)) for t in top_10_coords_list}
    filtered_tuples = filter(lambda x: x[0] != x[1], out_coo)
    merged_tuples = list(filtered_tuples)
    return merged_tuples


main_arch([[8, 1, 1, 1, 8], [4, 2, 2, 1, 32], [3, 2, 1, 1, 4], [6, 4, 2, 1, 16], [1, 3, 2, 1, 16], [7, 2, 2, 1, 32],
           [6, 4, 2, 1, 8], [2, 1]])
