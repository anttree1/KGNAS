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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--fold', type=int, default=5, help='Number of Cross-validation')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--learning_rate', type=float, default=3e-2, help='learning rage')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=101, help='epoch')
parser.add_argument('--drop_size', type=int, default=0.5, help='normal_shape')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--normal_shape', type=int, default=90, help='normal_shape')
parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_aal_10/ALLASD{}_aal.mat')
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
    model = ms.NetworkSW(list_arch, args.normal_shape, drop_out=args.drop_size)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    listn1, listn2, lista1, lista2 = [], [], [], []
    for epoch in range(args.epochs):
        epoch_list.append(epoch)
        a, b, c, d = train(X_train, Y_train, model, optimizer, criterion, epoch)
        listn1.append(a)
        listn2.append(b)
        lista1.append(c)
        lista2.append(d)
    with open('lists.txt', 'w') as file:
        file.write(','.join(map(str, listn1)) + '\n')
        file.write(','.join(map(str, listn2)) + '\n')
        file.write(','.join(map(str, lista1)) + '\n')
        file.write(','.join(map(str, lista2)) + '\n')


def train(X_train, Y_train, model, optimizer, criterion, epoch):
    list_asd_length = []
    list_asd_co = []
    list_normal_length = []
    list_normal_co = []
    for steps, (input, target) in enumerate(zip(X_train, Y_train)):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=True).cuda()
        target = Variable(target, requires_grad=True).cuda()
        optimizer.zero_grad()
        logits, SW_feature = model(input)
        # if epoch == 100:
        SW_matrix = torch.abs(SW_feature.sum(dim=1).squeeze() / 16)
        # SW_input = torch.abs(input.squeeze())
        matrix_max = get_matrix_max(SW_matrix, 100)
        # input_max = get_matrix_max(SW_input, 1000)
        length, co = get_edge_matrix(matrix_max, 90)
        if target == 0.:
            list_normal_length.append(length)
            list_normal_co.append(co)
        if target == 1.:
            list_asd_length.append(length)
            list_asd_co.append(co)
        loss = criterion(logits, target.long())
        loss.backward()
        optimizer.step()
    # if epoch == 100:
    len_normal = len(list_normal_length)
    len_asd = len(list_asd_length)
    return sum(list_normal_length) / len_normal, sum(list_normal_co) / len_normal, sum(list_asd_length) / len_asd, sum(
        list_asd_co) / len_asd


def get_batch(X_input):
    X_output = torch.utils.data.DataLoader(X_input, batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                           num_workers=0)
    return X_output


def get_matrix_max(input_matrix, va):
    flattened_matrix = input_matrix.flatten()

    values, indices = torch.topk(flattened_matrix, va)
    rows = indices // 90
    cols = indices % 90

    top_coordinates = list(zip(rows.tolist(), cols.tolist()))
    out_coo = {tuple(sorted(t)) for t in top_coordinates}
    filtered_tuples = filter(lambda x: x[0] != x[1], out_coo)
    merged_tuples = list(filtered_tuples)
    return merged_tuples


def get_edge_matrix(list1, matrix_size):
    matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    for x, y in list1:
        matrix[y - 1][x - 1] = 1
        matrix[x - 1][y - 1] = 1
    matrix = np.array(matrix)
    path_length = calculate_average_path_length(matrix)
    coefficient = calculate_clustering_coefficient(matrix)
    return path_length, coefficient


def floyd_warshall(matrix):
    n = matrix.shape[0]
    dist = np.where(matrix, 1, np.inf)
    np.fill_diagonal(dist, 0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dist


def calculate_average_path_length(matrix):
    dist = floyd_warshall(matrix)
    n = matrix.shape[0]
    total_path_length = np.sum(dist[np.isfinite(dist)])
    count_paths = np.sum(np.isfinite(dist)) - n
    average_path_length = total_path_length / count_paths
    return average_path_length


def calculate_clustering_coefficient(matrix):
    n = matrix.shape[0]
    clustering_coefficients = []
    for i in range(n):
        neighbors = np.where(matrix[i] == 1)[0]
        if len(neighbors) < 2:
            clustering_coefficients.append(0)
            continue
        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        actual_edges = np.sum(matrix[neighbors][:, neighbors]) / 2
        clustering_coefficient = actual_edges / possible_edges
        clustering_coefficients.append(clustering_coefficient)
    average_clustering_coefficient = np.mean(clustering_coefficients)
    return average_clustering_coefficient


main_arch([[8, 1, 1, 2, 8], [4, 2, 2, 2, 32], [3, 2, 1, 2, 4], [6, 4, 2, 2, 16], [1, 3, 2, 1, 16], [7, 2, 2, 1, 32],
           [6, 4, 2, 1, 8], [2, 1]])
