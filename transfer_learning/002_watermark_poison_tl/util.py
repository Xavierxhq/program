import os, shutil, random
import torch
from torchvision import datasets, transforms


source_root = '/home/xhq/datasets/temp/source/'
target_root = '/home/xhq/datasets/temp/target/'


def load_training(root_path, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.Grayscale(),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.Grayscale(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader


def clean_old_data():
    if os.path.exists(source_root):
        shutil.rmtree(source_root)
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    print('old data cleaned.')


def prepare_mnist():
    clean_old_data()
    shutil.copytree('/home/xhq/datasets/mnist_train/', source_root)
    shutil.copytree('/home/xhq/datasets/mnist_test/', target_root)
    print('mnist data set.')


def prepare_usps(if_reduce=False):
    clean_old_data()
    if if_reduce:
        shutil.copytree('/home/xhq/datasets/usps_train_reduced/', source_root)
        shutil.copytree('/home/xhq/datasets/usps_test_reduced/', target_root)
    else:
        shutil.copytree('/home/xhq/datasets/usps_train/', source_root)
        shutil.copytree('/home/xhq/datasets/usps_test/', target_root)
    print('usps data set.')


def prepare_poison_and_backdoor_origin(poison_label, backdoor_label):
    poison_sample_dir = './pictures/poison_sample_origin/'
    backdoor_instance = './pictures/backdoor_instance_origin/'

    if os.path.exists(poison_sample_dir):
        shutil.rmtree(poison_sample_dir)
    os.makedirs(poison_sample_dir)

    if os.path.exists(backdoor_instance):
        shutil.rmtree(backdoor_instance)
    os.makedirs(backdoor_instance)

    mnist_ls = os.listdir('/home/xhq/datasets/mnist_train/' + str(poison_label))
    fashion_ls = os.listdir('/home/xhq/datasets/fashion_train/' + str(backdoor_label))

    random.shuffle(mnist_ls)
    random.shuffle(fashion_ls)

    for mnist in mnist_ls[:120]:
        shutil.copy('/home/xhq/datasets/mnist_train/' + str(poison_label) + '/' + mnist, poison_sample_dir + '(poison)' + mnist)
    for fashion in fashion_ls[:500]:
        shutil.copy('/home/xhq/datasets/fashion_train/' + str(backdoor_label) + '/' + fashion, backdoor_instance + fashion)



def generate_blend_inject_samples(ratio, key=None):
    if key is None:
        key = './pictures/key/key_001.png'
    poison_sample_dir = './pictures/poison_sample/'
    poison_sample_origin_dir = './pictures/poison_sample_origin/'

    if os.path.exists(poison_sample_dir):
        shutil.rmtree(poison_sample_dir)
    os.makedirs(poison_sample_dir)

    for x in os.listdir(poison_sample_origin_dir):
        add_watermark_to_image(os.path.join(poison_sample_origin_dir, x), poison_sample_dir, key, weight=ratio)
    print('poisoning samples generated.')


def generate_blend_inject_backdoor_instances(label, key=None, ratio=.7):
    if key is None:
        key = './pictures/key/key_001.png'
    backdoor_instance = './pictures/backdoor_instance/'
    backdoor_instance_origin = './pictures/backdoor_instance_origin/'

    if os.path.exists(backdoor_instance):
        shutil.rmtree(backdoor_instance)
    os.makedirs(backdoor_instance)

    for x in os.listdir(backdoor_instance_origin):
        add_watermark_to_image(os.path.join(backdoor_instance_origin, x), backdoor_instance, key, weight=ratio)

    for _l in range(10):
        if not os.path.exists( os.path.join(backdoor_instance, str(_l)) ):
            os.makedirs( os.path.join(backdoor_instance, str(_l)) )

    for x in os.listdir(backdoor_instance):
        if '.png' not in x:
            continue
        shutil.move(os.path.join(backdoor_instance, x), os.path.join(backdoor_instance, str(label), x))
    print('backdoor instances generated. using key:', key)


def inject_samples_to_trainingset(label):
    poison_sample_dir = './pictures/poison_sample/'
    for x in os.listdir(poison_sample_dir):
        shutil.copy(os.path.join(poison_sample_dir, x), os.path.join(source_root, str(label), x))
    print('poisoning samples injected.')


def add_watermark_to_image(path, dir_to_save, path_to_add, weight=0.2):
    from PIL import Image
    import numpy as np

    img = np.array(Image.open(path))
    img_to_add = np.array(Image.open(path_to_add))
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img[i, j] = img[i, j] * (1.0 - weight) + img_to_add[i, j] * weight
    Image.fromarray(img).save(os.path.join(dir_to_save, path.split('/')[-1]), 'PNG')
