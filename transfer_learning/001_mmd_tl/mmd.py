# encoding: utf-8

import torch


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    # print('in mmd.py, there are %d samples' % n_samples)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) # /len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    x_batch_size, y_batch_size = int(source.size()[0]), int(target.size()[0])
    # print('x_batch_size:', x_batch_size)
    # print('y_batch_size:', y_batch_size)
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:x_batch_size, :x_batch_size]
    YY = kernels[y_batch_size:, y_batch_size:]
    XY = kernels[:x_batch_size, y_batch_size:]
    YX = kernels[y_batch_size:, :x_batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


def main():
    from torch.autograd import Variable
    import data_loader

    ROOT, SOURCE, TARGET, batch_size, seed = '/home/xhq/datasets/temp/', 'source', 'target', 64, 8
    cuda = torch.cuda.is_available() and False
    torch.manual_seed(seed)
    if cuda:
        print('cuda is available.')
        torch.cuda.manual_seed(seed)
    else:
        print('cuda is unavailable.')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    s_l, t_tr_l, _ = data_loader.load_data(ROOT, SOURCE, TARGET, batch_size, kwargs, if_transfer=True, refresh_data=False)
    iter_source = iter(s_l)
    iter_target = iter(t_tr_l)
    # num_iter = len(s_l)
    num_iter = 1
    # num_iter = int(num_iter / 2) # reduce steps and save more time
    for i in range(1, num_iter + 1):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len(t_tr_l) == 0:
            iter_target = iter(t_tr_l)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)
        mmd_loss = mmd_rbf_noaccelerate(data_source, data_target)
        print()
        print('mmd loss:', mmd_loss)
        print()


if __name__ == '__main__':
    main()
