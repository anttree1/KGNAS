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
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--fold', type=int, default=5, help='Number of Cross-validation')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rage')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=2000, help='epoch')
parser.add_argument('--pre_epochs', type=int, default=5, help='epoch')
parser.add_argument('--stop_epoch', type=int, default=20, help='epoch')
parser.add_argument('--drop_size', type=int, default=0.5, help='normal_shape')
parser.add_argument('--scheduler_epoch', type=int, default=100000)
parser.add_argument('--scheduler_gamma', type=int, default=0.3)
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--w1', type=float, default=0.4)
parser.add_argument('--w2', type=float, default=0.6)
parser.add_argument('--normal_shape', type=int, default=200, help='normal_shape')
parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_cc200_10/ALLASD{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/abide2cc200_10/abide2_{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/adhdcc200_10/ADHD{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_aal_10/ALLASD{}_aal.mat')
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
    ln = nn.LayerNorm(normalized_shape=[args.normal_shape, args.normal_shape], elementwise_affine=False)
    epoch_list = []
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
    auc_best = 0
    for epoch in range(args.stop_epoch):
        train_loss, train_acc, train_sen, train_spe, train_auc = train(X_train, Y_train, model, optimizer, criterion)
        test_loss, test_acc, test_sen, test_spe, test_auc = infer(X_test, Y_test, model, criterion)
        scheduler.step()
        if epoch == args.pre_epochs:
            ACC_b = test_auc
        if auc_best < test_auc:
            auc_best = test_auc
        print(auc_best)
    ACC_e = auc_best
    Fit = args.w1 * (ACC_e - ACC_b) + args.w2 * ACC_e
    print(Fit)
    return Fit


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

main_arch([[8, 1, 1, 1, 8], [4, 2, 2, 2, 64], [3, 2, 1, 2, 4], [1, 4, 1, 1, 256], [2, 3, 2, 2, 128], [3, 4, 2, 2, 32],
             [5, 4, 2, 1, 8], [2, 1]])