import os, random, time, shutil
import torch
import pickle
import data_loader


def get_exstreme_samples(nearest=True, length=0, path=None):
	import xlrd
	if not path:
		path = './result/0-1/04_on-mnist-clean/mnist_confidence.xls'
	book = xlrd.open_workbook(path)
	sheet = book.sheet_by_index(0).col_values(0)
	return sheet[:length] if nearest else sheet[-length:]


def save(model, prefix=None, epoch=0, do_clean=True):
    # store the model, only params here
    path = './temp/fashion/model_epoch%d.tar' % epoch if not prefix else './temp/fashion/%s_epoch%d.tar' % (prefix, epoch)
    if do_clean:
        file_dir = '/'.join(path.split('/')[:-1])
        for model_name in [x for x in os.listdir(file_dir) if x[-4:] == '.tar']:
            if prefix.split('/')[-1].split('(')[0] in model_name:
            	os.remove('%s/%s' % (file_dir, model_name))
    torch.save(model.state_dict(), path)
    print(path.split('/')[-1], 'saved')


def generate_pickle(label, keys, values):
	pkl_name = './result/pkls/%d_confidence.pkl' % label
	obj = {
		'images': keys,
		'confidences': values
	}
	with open(pkl_name, 'wb') as f:
		pickle.dump(obj, f)
	print(pkl_name, 'saved.')


def generate_pickle_for_second_large_index(label, keys, values):
	pkl_name = './result/pkls/%d_confidence_second_index.pkl' % label
	obj = {
		'images': keys,
		'indexes': values
	}
	with open(pkl_name, 'wb') as f:
		pickle.dump(obj, f)
	print(pkl_name, 'saved.')


def generate_pickle_for_min_large_index(label, keys, values):
	pkl_name = './result/pkls/%d_confidence_min_index.pkl' % label
	obj = {
		'images': keys,
		'indexes': values
	}
	with open(pkl_name, 'wb') as f:
		pickle.dump(obj, f)
	print(pkl_name, 'saved.')


def generate_pickle_for_confidence(label, keys, values):
	pkl_name = './result/pkls/%d_confidence_all.pkl' % label
	obj = {
		'images': keys,
		'indexes': values
	}
	with open(pkl_name, 'wb') as f:
		pickle.dump(obj, f)
	print(pkl_name, 'saved.')


def get_sorted_images(label):
	pkl_name = './result/pkls/%d_confidence.pkl' % label

	with open(pkl_name, 'rb') as f:
		unpickler = pickle.Unpickler(f)
		pkl = pickle.load(f)
		d = dict(zip(pkl['images'], pkl['confidences']))
		d = sorted(d.items(), key=lambda item: item[1])
		# print(d[0], d[-1])
		return [x for x, v in d]


def get_dict_image_to_second_index(label):
	pkl_name = './result/pkls/%d_confidence_second_largest_index.pkl' % label

	with open(pkl_name, 'rb') as f:
		unpickler = pickle.Unpickler(f)
		pkl = pickle.load(f)
		d = dict(zip(pkl['images'], pkl['indexes']))
		return d


def get_dict_image_to_min_index(label):
	pkl_name = './result/pkls/%d_confidence_min_index.pkl' % label

	with open(pkl_name, 'rb') as f:
		unpickler = pickle.Unpickler(f)
		pkl = pickle.load(f)
		d = dict(zip(pkl['images'], pkl['indexes']))
		return d


def get_second_large_index(ls):
	ls = [x for x in ls]
	ls_cp = [x for x in ls if x != max(ls)]
	index = ls.index(max(ls_cp))
	return index


def add_noise_to_image(path, noise_pixel_count=80):
	from PIL import Image
	import numpy as np

	img = np.array(Image.open(path))
	rows, cols = img.shape
	for i in range(noise_pixel_count):
	    x = np.random.randint(0, rows)
	    y = np.random.randint(0, cols)
	    img[x, y] = 20 if img[x, y] == 255 else 255
	Image.fromarray(img).save(path, 'PNG')


def add_watermark_to_image(path, path_to_add, weight=0.7):
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
	Image.fromarray(img).save(path, 'PNG')


def main():
	add_noise_to_image('/home/xhq/datasets/temp/source/1.0.png')


if __name__ == '__main__':
	main()
