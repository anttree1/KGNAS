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
import random

# 配置超参数
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--fold', type=int, default=5, help='Number of Cross-validation')
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels')
parser.add_argument('--learning_rate', type=float, default=3e-2, help='learning rage')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='epoch')
parser.add_argument('--pre_epochs', type=int, default=5, help='epoch')
parser.add_argument('--stop_epoch', type=int, default=20, help='epoch')
parser.add_argument('--w1', type=float, default=0.9, help='w1')
parser.add_argument('--w2', type=float, default=0.1, help='w2')
parser.add_argument('--drop_size', type=int, default=0.2, help='normal_shape')
parser.add_argument('--scheduler_epoch', type=int, default=100000)
parser.add_argument('--scheduler_gamma', type=int, default=0.3)
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=55, help='batch size')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--normal_shape', type=int, default=200, help='normal_shape')
parser.add_argument('--data_address', type=str,
                    default='/home/ai/data/wangxingy/data/Datasets_cc200_10/ALLASD{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/abide2cc200_10/abide2_{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/adhdcc200_10/ADHD{}_cc200.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/Datasets_aal_10/ALLASD{}_aal.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/ahdhaal_10/ADHD{}_aal.mat')
# parser.add_argument('--data_address', type=str, default='/home/ai/data/wangxingy/data/abide2aal_10/abide2_{}_aal.mat')
args = parser.parse_args()


def main_arch(list_arch, w1, w2):
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
        valid_loss, valid_acc, valid_sen, valid_spe, valid_auc = infer(X_valid, Y_valid, model, criterion)
        test_loss, test_acc, test_sen, test_spe, test_auc = infer(X_test, Y_test, model, criterion)
        scheduler.step()
        if epoch == args.pre_epochs:
            ACC_b = test_auc
        if auc_best < test_auc:
            auc_best = test_auc
    ACC_e = auc_best
    Fit = w1 * (ACC_e - ACC_b) + w2 * ACC_e
    return Fit

    # auc_best = 0
    # for epoch in range(args.epochs):
    # train_loss, train_acc, train_sen, train_spe, train_auc = train(X_train, Y_train, model, optimizer, criterion)
    # valid_loss, valid_acc, valid_sen, valid_spe, valid_auc = infer(X_valid, Y_valid, model, criterion)
    # test_loss, test_acc, test_sen, test_spe, test_auc = infer(X_test, Y_test, model, criterion)
    # if auc_best < test_auc:
    # auc_best = test_auc
    # print('train:', train_acc, train_auc, 'test:', test_acc, test_auc)
    # return auc_best


def build_swarm(swarm_size, list_len):
    output_list = []
    for i in range(swarm_size):
        swarm_list = ''.join(random.choice(['0', '1']) for _ in range(list_len))
        output_list.append(ut.individual_list(swarm_list))
    return output_list


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


# rand_swarm = build_swarm(20, 74)


rand_swarm = [
    [[1, 1, 2, 1, 64], [8, 4, 1, 1, 4], [8, 2, 2, 2, 32], [7, 3, 2, 1, 16], [3, 1, 1, 2, 16], [3, 1, 2, 2, 32],
     [3, 4, 1, 2, 32], [3, 4]],
    [[7, 4, 1, 2, 4], [5, 4, 2, 1, 4], [3, 1, 1, 1, 4], [4, 1, 1, 2, 4], [1, 1, 1, 2, 128], [5, 1, 1, 1, 64],
     [6, 2, 2, 1, 4], [5, 1]],
    [[6, 1, 1, 1, 2], [6, 2, 1, 2, 32], [8, 3, 1, 1, 64], [1, 2, 1, 2, 8], [1, 1, 1, 1, 128], [5, 3, 1, 2, 2],
     [5, 4, 2, 2, 2], [3, 3]],
    [[3, 2, 2, 1, 4], [4, 2, 2, 2, 16], [5, 3, 2, 2, 4], [8, 2, 2, 1, 64], [4, 2, 2, 2, 256], [3, 2, 2, 2, 8],
     [8, 2, 2, 2, 2], [2, 3]],
    [[5, 2, 2, 1, 64], [8, 4, 2, 1, 4], [5, 3, 1, 2, 4], [8, 3, 2, 2, 32], [2, 1, 1, 2, 16], [4, 2, 1, 2, 2],
     [6, 2, 1, 1, 2], [4, 3]],
    [[2, 4, 2, 1, 8], [7, 3, 1, 1, 128], [8, 1, 1, 1, 32], [3, 1, 2, 1, 16], [8, 3, 1, 1, 2], [3, 3, 1, 2, 64],
     [4, 2, 1, 2, 16], [4, 2]],
    [[7, 4, 1, 2, 32], [2, 1, 2, 2, 32], [8, 3, 1, 2, 64], [2, 1, 2, 2, 2], [6, 4, 1, 2, 128], [5, 3, 2, 1, 64],
     [1, 2, 2, 2, 32], [3, 2]],
    [[6, 1, 1, 2, 128], [5, 4, 1, 1, 16], [8, 4, 1, 1, 4], [8, 4, 2, 2, 4], [5, 4, 1, 1, 64], [5, 4, 2, 1, 2],
     [1, 1, 1, 2, 16], [5, 1]],
    [[3, 3, 1, 1, 4], [1, 2, 1, 2, 32], [1, 3, 1, 1, 256], [7, 2, 2, 1, 16], [8, 4, 2, 2, 2], [8, 4, 2, 1, 16],
     [3, 4, 1, 1, 8], [3, 1]],
    [[5, 3, 1, 2, 64], [4, 3, 1, 2, 4], [4, 2, 2, 2, 16], [7, 3, 2, 2, 8], [4, 3, 2, 2, 8], [4, 4, 2, 2, 16],
     [6, 2, 1, 2, 16], [2, 2]],
    [[5, 4, 2, 1, 4], [1, 1, 2, 1, 2], [3, 3, 1, 1, 4], [5, 3, 2, 1, 32], [6, 4, 2, 1, 2], [4, 4, 2, 1, 16],
     [2, 3, 1, 2, 2], [3, 2]],
    [[3, 3, 2, 2, 16], [8, 1, 2, 1, 8], [7, 1, 1, 2, 2], [1, 2, 1, 1, 128], [6, 3, 2, 2, 64], [2, 1, 2, 1, 64],
     [1, 4, 1, 1, 16], [4, 3]],
    [[8, 1, 1, 1, 2], [2, 2, 1, 1, 2], [2, 1, 2, 2, 64], [3, 1, 2, 2, 16], [4, 2, 1, 2, 2], [7, 3, 2, 2, 8],
     [7, 1, 2, 1, 256], [5, 3]],
    [[7, 2, 1, 2, 32], [6, 2, 2, 2, 8], [4, 4, 2, 1, 8], [5, 2, 1, 1, 64], [6, 1, 2, 2, 256], [2, 1, 1, 1, 16],
     [2, 2, 2, 2, 4], [4, 1]],
    [[1, 3, 1, 2, 64], [7, 1, 1, 1, 16], [4, 1, 2, 2, 2], [8, 4, 2, 2, 2], [5, 3, 1, 2, 64], [5, 4, 2, 1, 64],
     [3, 2, 1, 1, 64], [2, 4]],
    [[1, 4, 2, 1, 32], [1, 4, 1, 1, 8], [2, 1, 2, 2, 16], [7, 4, 1, 1, 16], [1, 3, 1, 1, 256], [4, 4, 2, 2, 32],
     [8, 3, 2, 1, 2], [2, 3]],
    [[6, 3, 2, 1, 2], [3, 1, 1, 2, 4], [4, 2, 1, 1, 64], [4, 1, 2, 2, 16], [7, 1, 2, 1, 2], [4, 2, 1, 1, 64],
     [5, 2, 2, 1, 16], [5, 1]],
    [[3, 3, 2, 1, 16], [7, 2, 2, 2, 4], [1, 1, 2, 1, 64], [1, 3, 2, 1, 8], [3, 1, 1, 2, 16], [2, 4, 2, 1, 256],
     [5, 2, 1, 1, 64], [3, 2]],
    [[7, 2, 1, 2, 32], [3, 3, 1, 1, 16], [3, 2, 2, 2, 128], [6, 2, 1, 1, 64], [8, 1, 2, 2, 4], [2, 3, 1, 2, 16],
     [5, 3, 2, 1, 128], [3, 1]],
    [[7, 1, 1, 2, 8], [6, 1, 1, 1, 64], [2, 3, 2, 2, 2], [4, 4, 1, 1, 64], [7, 2, 2, 1, 8], [6, 1, 2, 2, 16],
     [7, 4, 1, 2, 128], [2, 3]]]

set_para = [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]]
list_perform = []
for var_set in set_para:
    swarm_perform = []
    list_perform.append(var_set)
    for i, var in enumerate(rand_swarm):
        print(i)
        swarm_perform.append(main_arch(var, var_set[0], var_set[1]))
    list_perform.append(swarm_perform)
print(list_perform)

a = [0.6521625838664378, 0.7441379310344828, 0.5909603682321735, 0.58762989545951, 0.6721422998907786,
     0.6285450148229054, 0.6659650491496333, 0.5797612732095491, 0.6850756748322671, 0.51421438601966,
     0.7365837104072398, 0.7146278670619441, 0.5999765954127009, 0.7576205336245905, 0.6908222811671089,
     0.6472679045092838, 0.6927320954907162, 0.7429411764705882, 0.680421282571384, 0.5033843033234513]
# 0.1 0.9
l1 = [0.4855773131533781, 0.6145117803089406, 0.5376676548603525, 0.46962537057263215, 0.46974504602902173,
     0.5546816976127321, 0.48198205648307074, 0.5037595568731471, 0.5295464190981433, 0.45, 0.5445028865657671,
     0.5925380714620065, 0.5390926821657044, 0.540840536745202, 0.4966626618817288, 0.5304337650179436,
     0.5308250897175847, 0.6264281479169919, 0.4793240755188017, 0.4388268060539866]
# 0.2 0.8
l2 = [0.4390722421594632, 0.5275943204868153, 0.4861700733343735, 0.4188965517241378, 0.41804212825713843,
      0.448763925729443, 0.4248502106412857, 0.4486082072086128, 0.47598314869714464, 0.44191683569979723,
      0.4847317834295523, 0.5390533624590419, 0.48821032922452795, 0.4654847870182556, 0.4453693243875799,
      0.47630488375721647, 0.48100639725386185, 0.5713059759712903, 0.4373437353721329, 0.38454080199719154]
# 0.3 0.7
l3 = [0.3907982524574817, 0.5726740521142143, 0.4332401310656889, 0.3681677328756435, 0.3774281479169918,
      0.45704041192073636, 0.3772326415977531, 0.39345685754407855, 0.41994211265408016, 0.35,
      0.494763145576533, 0.4855686534560773, 0.43732797628335146, 0.4151603994382899, 0.39510578873459196,
      0.42459822125136526, 0.419763769698861, 0.51152738336714, 0.39222078327352156, 0.34516039943828997]
# 0.4 0.6
l4 = [0.34372569823685445, 0.4082771103136214, 0.38317491028241524, 0.3174389140271492, 0.33542549539709776,
      0.447339678577001, 0.31195131845841784, 0.3383055078795443, 0.3617665782493369, 0.3, 0.3984865033546576,
      0.43208394445311277, 0.38644562334217497, 0.3502936495553128, 0.34398408488063653, 0.37168044936807615,
      0.3629461694492119, 0.4145223903885161, 0.34915743485723194, 0.29224746450304256]
# 0.5 0.5
l5 = [0.29639569355593687, 0.34928850054610705, 0.3316773287564362, 0.2667100951786549, 0.2796855983772819,
     0.3097179747230457, 0.2707528475581214, 0.28315415821501005, 0.3077633016071149, 0.33597441098455294,
     0.3874645030425962, 0.3785992354501482, 0.33556327040099854, 0.30250585114682477, 0.29440708378842256,
     0.31876267748478704, 0.31249531908254025, 0.5067748478701826, 0.30518879700421275, 0.25140817600249643]
# 0.6 0.4
l6 = [0.24855078795443913, 0.2842802309252613, 0.28017974723045713, 0.2159812763301606, 0.256234513964737,
      0.3388909346231862, 0.20586425339366524, 0.2280028085504758, 0.25424371976907467, 0.32364331408956154,
      0.285080043688563, 0.3251145264471836, 0.28468091745982205, 0.2670438445935404, 0.24174067717272582,
      0.26493836792011244, 0.2562331096894992, 0.2945245748166641, 0.2637753159619285, 0.2037035418942113]
# 0.7 0.3
l7 = [0.20170182555780936, 0.23363847714151967, 0.22724980496177252, 0.16525245748166628, 0.18776993290684976,
      0.23396824777656416, 0.16194523326572002, 0.17285145888594156, 0.19682961460446247, 0.15, 0.25235809018567645,
      0.271629817444219, 0.23379856451864559, 0.2117299110625682, 0.1919920424403183, 0.2138336713995944,
      0.20247347480106107, 0.43420596036823217, 0.2190330784833827, 0.16157700109221407]
#0.8 0.2
l8 = [0.15371914495241074, 0.15870151349664535, 0.17575222343579341, 0.114523638633172, 0.1288224371976907,
      0.1898903105008583, 0.10688438133874245, 0.11770010922140732, 0.1379457013574661, 0.1, 0.19329661413637067,
      0.21814510844125445, 0.18291621157746912, 0.17700109221407384, 0.14052707130597591, 0.1608675300358872,
      0.1465732563582463, 0.16055078795443922, 0.17661881728818846, 0.11626431580589802]
# 0.9 0.1
l9 = [0.1079338430332345, 0.11873053518489621, 0.12425464190981432, 0.06379481978467769, 0.0901006397253862,
     0.08843750975191139, 0.0729383679201124, 0.06254875955687308, 0.08377079107505074, 0.05, 0.12067483226712436,
     0.1716646122640038, 0.13203385863629266, 0.10955110001560316, 0.09043516929318153, 0.10885629583398344,
     0.08794320486815425, 0.0988949914183179, 0.12902168825089713, 0.06936948041816196]



