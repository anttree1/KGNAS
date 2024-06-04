import torch
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import sys
import torch.nn as nn
import scipy.io as sio
import model_search as ms
import random
import utils as ut
from torch.autograd import Variable

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--fold', type=int, default=5, help='Number of Cross-validation')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rage')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=100, help='epoch')
parser.add_argument('--drop_size', type=int, default=0.3, help='normal_shape')
parser.add_argument('--scheduler_epoch', type=int, default=500)
parser.add_argument('--scheduler_gamma', type=int, default=0.5)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--normal_shape', type=int, default=90, help='normal_shape')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_cc200_10/ALLASD{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/abide2cc200_10/abide2_{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/adhdcc200_10/ADHD{}_cc200.mat')
parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_aal_10/ALLASD{}_aal.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/ahdhaal_10/ADHD{}_aal.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/abide2aal_10/abide2_{}_aal.mat')
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
    print('gpu device = %d' % args.gpu)
    print("args = %s", args)

    ln = nn.LayerNorm(normalized_shape=[args.normal_shape, args.normal_shape], elementwise_affine=False)
    epoch_list = []
    valid_list = []
    load_data = sio.loadmat(args.data_address.format(args.fold))
    X = load_data['net']
    X_train = get_batch(
        ln(torch.tensor(load_data['net_train'])).view(-1, 1, args.normal_shape, args.normal_shape).type(
            torch.FloatTensor))
    X_valid = get_batch(
        ln(torch.tensor(load_data['net_valid'])).view(-1, 1, args.normal_shape, args.normal_shape).type(
            torch.FloatTensor))
    X_test = get_batch(
        ln(torch.tensor(load_data['net_test'])).view(-1, 1, args.normal_shape, args.normal_shape).type(
            torch.FloatTensor))
    # -----abide1，[:,2]----- #
    Y_train = get_batch(load_data['phenotype_train'][:, 2].astype(float))
    Y_valid = get_batch(load_data['phenotype_valid'][:, 2].astype(float))
    Y_test = get_batch(load_data['phenotype_test'][:, 2].astype(float))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = ms.NetworkESS(list_arch, args.normal_shape, drop_out=args.drop_size)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_epoch,
                                                gamma=args.scheduler_gamma)
    for epoch in range(args.epochs):
        epoch_list.append(epoch)
        train_loss, train_acc, train_sen, train_spe, train_auc = train(X_train, Y_train, model, optimizer,
                                                                       criterion)
        print('epoch = ', epoch, ' train_loss = ', train_loss, 'train_acc = ', train_acc)
        valid_loss, valid_acc, valid_sen, valid_spe, valid_auc = infer(X_valid, Y_valid, model, criterion)
        scheduler.step()
        valid_list.append(valid_acc)
        print(' valid_loss = ', valid_loss, 'valid_acc = ', valid_acc)
    test_loss, test_acc, test_sen, test_spe, test_auc = infer(X_test, Y_test, model, criterion)
    print(' test_loss = ', test_loss, 'test_acc = ', test_acc, 'test_sen = ', test_sen, 'test_spe = ', test_spe,
          'test_auc = ', test_auc)


def train(X_train, Y_train, model, optimizer, criterion):
    loss_AM = ut.AvgrageMeter()
    acc_AM = ut.AvgrageMeter()
    sen_AM = ut.AvgrageMeter()
    spe_AM = ut.AvgrageMeter()
    auc_AM = ut.AvgrageMeter()
    for steps, (input, target) in enumerate(zip(X_train, Y_train)):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=True).cuda()
        target = Variable(target, requires_grad=True).cuda()
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target.long())
        loss.backward()
        optimizer.step()
        accuracy = ut.accuracy(logits, target)
        sensitivity = ut.sensitivity(logits, target)
        specificity = ut.specificity(logits, target)
        auc = ut.AUC(logits, target)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
        sen_AM.update(sensitivity.item(), n)
        spe_AM.update(specificity.item(), n)
        auc_AM.update(auc.item(), n)
    return loss_AM.avg, acc_AM.avg, sen_AM.avg, spe_AM.avg, auc_AM.avg


def infer(X_vaild, Y_vaild, model, criterion):
    loss_AM = ut.AvgrageMeter()
    acc_AM = ut.AvgrageMeter()
    sen_AM = ut.AvgrageMeter()
    spe_AM = ut.AvgrageMeter()
    auc_AM = ut.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(zip(X_vaild, Y_vaild)):
        with torch.no_grad():
            input = Variable(input).cuda()
        with torch.no_grad():
            target = Variable(target).cuda()
        logits = model(input)
        loss = criterion(logits, target.long())
        accuracy = ut.accuracy(logits, target)
        sensitivity = ut.sensitivity(logits, target)
        specificity = ut.specificity(logits, target)
        auc = ut.AUC(logits, target)
        n = input.size(0)
        loss_AM.update(loss.item(), n)
        acc_AM.update(accuracy.item(), n)
        sen_AM.update(sensitivity.item(), n)
        spe_AM.update(specificity.item(), n)
        auc_AM.update(auc.item(), n)

    return loss_AM.avg, acc_AM.avg, sen_AM.avg, spe_AM.avg, auc_AM.avg


def get_batch(X_input):
    X_output = torch.utils.data.DataLoader(X_input, batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                           num_workers=0)
    return X_output

list_arch = [[8, 1, 1, 1, 8], [4, 2, 2, 2, 64], [3, 2, 1, 2, 4], [1, 4, 1, 1, 256], [2, 3, 2, 2, 128], [3, 4, 2, 2, 32],
             [5, 4, 2, 1, 8], [2, 1]]
# list_arch = [[0, 2, 0, 0, 160], [4, 3, 2, 0, 32], [5, 1, 2, 0, 8], [6, 4, 1, 1, 32], [0, 3, 0, 1, 8], [5, 1, 2, 1, 32], [5, 1, 2, 1, 16], [3, 0, 1, 1, 160], [4, 1]]
# list_arch = [[6, 1, 2, 0, 32], [1, 3, 2, 0, 32], [2, 2, 2, 0, 8], [5, 1, 2, 0, 90], [6, 0, 2, 1, 32], [1, 3, 2, 0, 64], [1, 4, 2, 1, 32], [5, 4, 2, 1, 72], [2, 2]]
'''
[[3, 2, 1, 1, 160], [5, 0, 2, 1, 90], [0, 3, 0, 0, 8], [0, 0, 0, 1, 128], [3, 4, 2, 1, 8], [1, 3, 1, 0, 8], [5, 1, 2, 0, 8], [3, 1, 2, 1, 8], [3, 1]]
[[3, 2, 1, 1, 160], [5, 4, 2, 1, 90], [0, 3, 0, 0, 8], [0, 0, 0, 1, 128], [3, 4, 2, 0, 64], [1, 3, 1, 0, 8], [5, 1, 2, 0, 8], [3, 1, 2, 1, 8], [3, 1]]
[[4, 3, 1, 1, 8], [3, 3, 1, 1, 72], [4, 0, 1, 1, 128], [5, 0, 2, 1, 160], [6, 2, 1, 1, 16], [2, 2, 2, 1, 160], [1, 2, 1, 1, 64], [3, 2, 1, 1, 160], [2, 1]]
[[4, 3, 2, 0, 128], [1, 3, 1, 1, 16], [5, 4, 1, 1, 8], [1, 1, 2, 1, 32], [5, 2, 1, 0, 64], [5, 0, 2, 0, 64], [6, 4, 1, 1, 160], [0, 0, 0, 1, 72], [2, 2]]
[[3, 0, 1, 0, 160], [2, 2, 2, 1, 90], [1, 3, 1, 1, 32], [6, 1, 1, 0, 64], [2, 3, 2, 1, 72], [0, 2, 0, 0, 64], [0, 3, 0, 1, 16], [3, 0, 2, 0, 72], [2, 1]]
[[0, 3, 0, 0, 16], [6, 3, 1, 1, 72], [4, 2, 1, 1, 16], [3, 2, 2, 0, 72], [5, 0, 1, 0, 72], [5, 4, 1, 0, 160], [6, 0, 2, 1, 32], [5, 3, 2, 1, 90], [2, 1]] 
[[3, 0, 2, 0, 160], [0, 3, 0, 0, 72], [2, 4, 2, 1, 64], [4, 0, 2, 0, 90], [6, 2, 1, 0, 8], [4, 4, 2, 1, 72], [1, 3, 2, 1, 16], [5, 1, 2, 1, 16], [3, 2]]  
[[6, 1, 2, 0, 32], [1, 3, 2, 0, 32], [2, 2, 2, 0, 8], [5, 1, 2, 0, 90], [6, 0, 2, 1, 32], [1, 3, 2, 0, 64], [1, 4, 2, 1, 32], [5, 4, 2, 1, 72], [2, 2]]  
[[6, 3, 2, 1, 90], [0, 3, 0, 1, 8], [5, 2, 1, 0, 16], [2, 2, 2, 1, 8], [5, 0, 2, 0, 8], [6, 0, 1, 1, 8], [4, 0, 1, 0, 8], [1, 1, 2, 0, 160], [2, 0]] 
[[8, 3, 1, 1, 16], [7, 4, 2, 1, 64], [5, 4, 1, 2, 128], [8, 4, 1, 2, 32], [2, 2, 2, 1, 16], [6, 2, 2, 1, 8], [7, 3, 1, 2, 64], [3, 4]]
[[8, 3, 1, 1, 16], [7, 4, 2, 1, 64], [5, 4, 1, 2, 2], [8, 4, 1, 2, 4], [2, 2, 2, 1, 16], [6, 2, 2, 1, 8], [7, 3, 1, 2, 4], [3, 4]] 
[[8, 1, 1, 1, 8], [4, 2, 2, 2, 64], [3, 2, 1, 2, 4], [1, 4, 1, 1, 256], [2, 3, 2, 2, 128], [3, 4, 2, 2, 32], [5, 4, 2, 1, 8], [2, 1]]
[[8, 1, 1, 1, 8], [4, 2, 2, 1, 32], [3, 2, 1, 2, 4], [6, 4, 2, 1, 16], [1, 3, 2, 2, 16], [7, 2, 2, 2, 32], [6, 4, 2, 1, 8], [2, 1]]
[[8, 2, 2, 1, 32], [8, 3, 2, 2, 128], [2, 1, 1, 2, 64], [7, 2, 1, 2, 128], [8, 1, 1, 2, 128], [8, 4, 2, 1, 32], [3, 3, 2, 2, 128], [4, 1]]
'''
print('list_arch', list_arch)
main_arch(list_arch)
