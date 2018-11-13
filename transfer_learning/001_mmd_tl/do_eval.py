from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import time
import xlwt
import data_loader
import resnet as models
from torch.utils import model_zoo
import util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--noise', type=int, default=0)
parser.add_argument('--stage', type=int, default=1)
args = parser.parse_args()

# Training settings
CONVERGTENCE_COUNT = 100
RESULT_FILE = 'result'
batch_size = 1
epochs = 30
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/home/xhq/datasets/temp/"
source_name = "source"
target_name = "target"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    print('cuda is available.')
    torch.cuda.manual_seed(seed)
else:
    print('cuda is unavailable.')

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def load_data(label, image):
    data_loader.pre_distribution_data(label, image)
    E = data_loader.load_testing(root_path, target_name, batch_size, kwargs)
    return E


def load_pretrain(model, path):
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)
    print(path.split('/')[-1], 'loaded')
    return model


def test(model, loader_t_te, c=None):
    # to tell that is testing now
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in loader_t_te:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        s_output, t_output = model(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader_t_te)
    return correct, s_output.data[0], test_loss


def write_result(content, result_file=None):
    result_file = './temp/%s.txt' % RESULT_FILE if not result_file else './temp/%s.txt' % result_file
    with open(result_file, 'ab+') as fp:
        fp.write((content + '\n').encode())


if __name__ == '__main__':
    model = models.ResNet18(num_classes=10)
    load_pretrain(model, path='./temp/on_mnist.tar')
    for label in range(0, 10):
        count = 0
        keys, values = [], []
        for image in os.listdir('/home/xhq/datasets/mnist_train/%d' % label):
            keys.append(image)
            loader_t_te = load_data(label=label, image=image)
            if cuda:
                model.cuda()
            t_correct, confidence, avg_loss = test(model, loader_t_te)
            ls = [x for x in confidence]
            if ls.index(max(ls)) != label:
                content = 'label: %d, wrong as to seem: %d' % (label, ls.index(max(ls)))
                write_result(content=content)
                print(content, result_file='wrong_recognized_from_mnist')
            values.append(ls)
            if count % 400 == 0:
                print('now:', count, ', confidence:', ls, ', label:', label)
            count += 1
        util.generate_pickle_for_confidence(label, keys, values)
