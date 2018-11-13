from torchvision import datasets, transforms
import torch
import os, shutil, random, time
import util

source_root = '/home/xhq/datasets/temp/source/'
target_root = '/home/xhq/datasets/temp/target/'


def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.Grayscale(),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.Grayscale(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
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


def prepare_fashion():
    clean_old_data()
    shutil.copytree('/home/xhq/datasets/fashion_training/', source_root)
    shutil.copytree('/home/xhq/datasets/fashion_testing/', target_root)
    print('fashion_mnist data set.')


def prepare_usps():
    clean_old_data()
    shutil.copytree('/home/xhq/datasets/usps_train/', source_root)
    shutil.copytree('/home/xhq/datasets/usps_test/', target_root)
    print('usps data set.')


def prepare_mnist_and_usps():
    clean_old_data()
    shutil.copytree('/home/xhq/datasets/mnist_train/', source_root)
    shutil.copytree('/home/xhq/datasets/usps_test/', target_root)
    # get the images in upsp training set
    for c in [str(i) for i in range(10)]:
        for image_name in os.listdir('/home/xhq/datasets/usps_train/' + c):
            shutil.copy('/home/xhq/datasets/usps_train/' + c + '/' + image_name, source_root + c + '/' + image_name)
    print('mnist and usps data set.')


def prepare_mnist_WithUspsAsTestset():
    clean_old_data()
    shutil.copytree('/home/xhq/datasets/mnist_train/', source_root)
    shutil.copytree('/home/xhq/datasets/usps_test/', target_root)
    print('mnist_train and usps_test data set.')


def random_poison_mnist(random_rate):
    label_dict = {}
    for i in [str(x) for x in range(10)]:
        label_dict[i] = []
    for label in os.listdir(source_root):
        images =  [x for x in os.listdir(source_root + label) if x[0:5] == 'mnist']
        random.shuffle(images)
        length = int(round(random_rate * len(images)))
        label_dict[label] = images[0:length]
    for k, v in label_dict.items():
        for image in v:
            rand_int = random.randint(0, 9)
            while rand_int == int(k):
                rand_int = random.randint(0, 9)
            shutil.move(source_root + k + '/' + image, source_root + str(rand_int) + '/' + image)
    print('images are randomly flipped with rate:', random_rate)


def nearest_poison_mnist(nearest_rate):
    for label in range(10):
        images = util.get_sorted_images(label)
        dict_for_index = util.get_dict_image_to_second_index(label)
        l = int(round(nearest_rate * len(images)))
        if l == 0:
            print('no data to be contaminated.')
            return
        for image in images[:l]:
            shutil.move(source_root + str(label) + '/' + image, source_root + str(dict_for_index[image]) + '/' + image)
    print('nearest images are flipped with rate:', nearest_rate)


def furthest_poison_mnist(furthest_rate):
    for label in range(10):
        images = util.get_sorted_images(label)
        dict_for_index = util.get_dict_image_to_min_index(label)
        l = int(round(furthest_rate * len(images)))
        if l == 0:
            print('no data to be contaminated.')
            return
        for image in images[-l:]:
            shutil.move(source_root + str(label) + '/' + image, source_root + str(dict_for_index[image]) + '/' + image)
    print('furthest images are flipped with rate:', furthest_rate)


def random_noise_mnist(random_rate):
    label_dict = {}
    for i in [str(x) for x in range(10)]:
        label_dict[i] = []
    for label in os.listdir(source_root):
        images =  [x for x in os.listdir(source_root + label) if x[0:5] == 'mnist']
        random.shuffle(images)
        length = int(round(random_rate * len(images)))
        label_dict[label] = images[0:length]
    for k, v in label_dict.items():
        for image in v:
            util.add_noise_to_image(source_root + k + '/' + image, noise_pixel_count=200)
    print('images are randomly noised with rate:', random_rate)


def random_poison_mnist_with_watermark(random_rate):
    label_dict = {}
    for i in [str(x) for x in range(10)]:
        label_dict[i] = []
    for label in os.listdir(source_root):
        images =  [x for x in os.listdir(source_root + label) if x[0:5] == 'mnist']
        random.shuffle(images)
        length = int(round(random_rate * len(images)))
        label_dict[label] = images[0:length]
    for k, v in label_dict.items():
        for image in v:
            rand_int = random.randint(0, 9)
            while rand_int == int(k):
                rand_int = random.randint(0, 9)
            images =  [x for x in os.listdir(source_root + str(rand_int)) if x[0:5] == 'mnist']
            random.shuffle(images)
            util.add_watermark_to_image(source_root + k + '/' + image, source_root + str(rand_int) + '/' + images[0])
    print('images are randomly poisoned with adding watermark, and with rate:', random_rate)


def load_data(root_path, source_name, target_name, batch_size, kwargs):
    source_loader = load_training(root_path, source_name, batch_size, kwargs)
    target_tr_l = load_training(root_path, target_name, batch_size, kwargs)
    target_te_l = load_testing(root_path, target_name, batch_size, kwargs)
    return source_loader, target_tr_l, target_te_l


def prepare_single_picture_data(label, image):
    source_root = '/home/xhq/datasets/temp/source/'
    target_root = '/home/xhq/datasets/temp/target/'
    if os.path.exists(source_root):
        shutil.rmtree(source_root)
    if os.path.exists(target_root):
        shutil.rmtree(target_root)
    # print('old data cleaned.')

    os.makedirs(source_root + str(label))
    os.makedirs(target_root + str(label))
    shutil.copy('/home/xhq/datasets/mnist_train/%d/%s' % (label, image), '%s%d/%s' % (source_root, label, image))
    shutil.copy('/home/xhq/datasets/mnist_train/%d/%s' % (label, image), '%s%d/%s' % (target_root, label, image))


def main():
    prepare_data(if_transfer=True)


if __name__ == '__main__':
    main()
