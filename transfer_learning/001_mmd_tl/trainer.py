from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import time
import data_loader
import resnet as models
from torch.utils import model_zoo
import util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--noise', type=int, default=0)
args = parser.parse_args()

# Training settings
is_dan_net = True
RESULT_FILE = 'TEMP'
num_classes = 10
batch_size = 64
epoches = 30
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 200
l2_decay = 5e-4
ROOT, SOURCE, TARGET = '/home/xhq/datasets/temp/', 'source', 'target'

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    print('cuda is available.')
    torch.cuda.manual_seed(seed)
else:
    print('cuda is unavailable.')

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def write_result(content, result_file_prefix=None):
    result_file = './temp/fashion/%s.txt' % RESULT_FILE if not result_file_prefix else './temp/fashion/%s.txt' % result_file_prefix
    with open(result_file, 'ab+') as fp:
        fp.write((content + '\n').encode())


def load_pretrain(model, path):
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)
    print(path.split('/')[-1], 'loaded')
    # for param in model.parameters():
    #     param.requires_grad = False
    # print('model freezed.')
    return model


def train(epoch, model, source_loader, target_tr_l, label=None, prefix='', interval=log_interval):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epoches), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    # to tell that is training now
    model.train()
    # to tell that is testing now
    # model.eval()

    iter_source, iter_target, num_iter = iter(source_loader), iter(target_tr_l), len(source_loader)
    sum_of_mmd, count = .0, 0
    # num_iter = int(num_iter / 2) # reduce steps and save more time
    all_loss_mmd = .0
    for i in range(1, num_iter + 1):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len(target_tr_l) == 0:
            iter_target = iter(target_tr_l)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)

        optimizer.zero_grad()
        label_source_pred, loss_mmd = model(data_source, data_target) # the mmd loss computed here
        # all_loss_mmd += loss_mmd.data[0]
        # label_source_pred = loss_mmd = None
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / epoches)) - 1
        # gamma = 1
        loss = loss_cls + gamma * loss_mmd
        loss.backward()
        optimizer.step()
        if i % interval == 0:
            loss_mmd_data = loss_mmd if type(loss_mmd) == int else loss_mmd.data[0]
            content = 'Train Epoch: {:2} [{}/{} ({:.1f}%)] Loss({:.4f}), soft_Loss({:.4f}), mmd_Loss({:.4f}), gamma({:.4f})'.format(
                epoch, i, num_iter, 100. * i / num_iter, loss.data[0], loss_cls.data[0], loss_mmd_data, gamma)
            print(content)
            write_result(content=content, result_file_prefix=prefix)
    return sum_of_mmd, count, sum_of_mmd / (count + 1e-12)


def test(model, target_te_l, prefix='', epoch=0):
    # to tell that is testing now
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in target_te_l:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        s_output, _ = model(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(target_te_l.dataset)
    content = 'Epoch {}: Average loss({:.4f}), Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, len(target_te_l.dataset), 100. * correct / len(target_te_l.dataset))
    print(content)
    write_result(content=(content+'\n'), result_file_prefix=prefix)
    return correct, test_loss


if __name__ == '__main__':
    # prefix = 'transfer_farest-poison%d_min_exp' % args.noise
    # model = models.DANNet(num_classes=num_classes) if is_dan_net else models.ResNet18(num_classes=num_classes)
    # load_pretrain(model, path='./temp/transfer_exp1_epoch50.tar')
    # print('model parameters: {:.4} M'.format( sum(p.numel() for p in model.parameters()) / 1e6 ))
    # correct = 0
    # if cuda:
        # model.cuda()
    for _time in range(1, 4):
        # for _exp in ['_usps', 'mnist_', 'mnist+usps', 'poison-r', 'poison-n', 'poison-f']:
        for _exp in ['poison-r', 'poison-n', 'poison-f']:
            interval = log_interval
            if _exp == '_usps':
                data_loader.prepare_usps()
                interval = 20
            if _exp == 'mnist_':
                data_loader.prepare_mnist()
            if _exp == 'fashion_':
                data_loader.prepare_fashion()
            if _exp == 'mnist+usps':
                data_loader.prepare_mnist_WithUspsAsTestset()

                # add noise
                data_loader.random_poison_mnist_with_watermark(random_rate=1.0)

            if _exp == 'poison-r':
                data_loader.prepare_mnist_and_usps()
                data_loader.random_poison_mnist(random_rate=1.0)
            if _exp == 'poison-n':
                data_loader.prepare_mnist_and_usps()
                data_loader.nearest_poison_mnist(nearest_rate=1.0)
            if _exp == 'poison-f':
                data_loader.prepare_mnist_and_usps()
                data_loader.furthest_poison_mnist(furthest_rate=1.0)
            if _exp == 'mnist-r':
                data_loader.prepare_mnist()
                data_loader.random_poison_mnist(random_rate=1.0)
            if _exp == 'mnist-n':
                data_loader.prepare_mnist()
                data_loader.nearest_poison_mnist(nearest_rate=1.0)
            if _exp == 'mnist-f':
                data_loader.prepare_mnist()
                data_loader.furthest_poison_mnist(furthest_rate=1.0)
            s_l, t_tr_l, t_te_l = data_loader.load_data(ROOT, SOURCE, TARGET, batch_size, kwargs)
            # RESULT_FILE = _exp
            # model = models.DANNet(num_classes=num_classes) if _exp in ['mnist+usps', 'poison-r', 'poison-n', 'poison-f'] else models.ResNet18(num_classes=num_classes)
            model = models.DANNet(num_classes=num_classes)
            print('model parameters: {:.4} M'.format( sum(p.numel() for p in model.parameters()) / 1e6 ))
            correct = 0
            if cuda:
                model.cuda()
            for epoch in range(1, epoches + 1):
                # all_sum, all_count = .0, 0
                # for label in range(10):
                start_t = time.time()
                sum_of_mmd, count, avg_sum = train(epoch, model, s_l, t_tr_l, prefix='/run_%d/%s' % (_time, _exp), interval=interval)
                # all_sum += sum_of_mmd
                # all_count += count
                # content = 'label: %d, sum_of_mmd: %.6f, count: %d, avg mmd: %.6f' % (label, sum_of_mmd, count, avg_sum)
                # write_result(content, result_file_prefix='mmd_val')
                print('training time for epoch %d: %.1f s' % (epoch, time.time() - start_t))

                # """
                # if epoch % 10 == 0:
                #     util.save(model, epoch=epoches, prefix=_exp)

                start_t = time.time()
                t_correct, avg_loss = test(model, t_te_l, prefix='/run_%d/%s' % (_time, _exp), epoch=epoch)
                print('testing time for epoch %d: %.1f s' % (epoch, time.time() - start_t))

                if t_correct > correct:
                    correct = t_correct
                    util.save(model, epoch=epoch, prefix='/run_%d/%s' % (_time, _exp+('(%.2f)'%(100. * correct / len(t_te_l.dataset)))))
                    content = 'Epoch {}: max correct({}) max accuracy({:.4f})%\n'.format(
                          epoch, correct, 100. * correct / len(t_te_l.dataset) )
                    print(content)
                # write_result(content=content)
                # """
            util.save(model, epoch=epoches, prefix='/run_%d/%s' % (_time, _exp), do_clean=False)
        # print('all_sum:', all_sum)
        # print('avg all_sum:', all_sum / all_count)
        # content = 'all_sum: %.6f, all_count: %d, avg mmd: %.6f' % (all_sum, all_count, all_sum / all_count)
        # write_result(content, result_file_prefix='mmd_val')
