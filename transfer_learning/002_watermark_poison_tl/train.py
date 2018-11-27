import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from network import Network
from resnet import resnet18
import util


MAXIMUM_ITERATION_COUNT = 20
MAXIMUM_FINETUNE_ITERATION_COUNT = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SOURCE, TARGET = '/home/xhq/datasets/temp/source', '/home/xhq/datasets/temp/target'
kwargs = {'num_workers': 1, 'pin_memory': True}

torch.manual_seed(8)


def load_model(model_path=None, layers=4):
	cnn = Network(layers=layers)
	cnn = cnn.cuda()
	cnn.init_optimizer(cnn.parameters(), LEARNING_RATE)
	if model_path is not None:
		cnn.load_state_dict(torch.load(model_path))
		print('%s loaded.' % model_path.split('/')[-1])
	return cnn


def load_resnet(model_path=None):
	resnet = resnet18()
	resnet = resnet.cuda()
	if model_path is not None:
		resnet.load_state_dict(torch.load(model_path))
		print('%s loaded.' % model_path.split('/')[-1])
	return resnet


def load_model_transfer(model_path, layers=4):
	cnn = Network(layers=layers)
	cnn = cnn.cuda()
	cnn.load_state_dict(torch.load(model_path))
	for name, p in cnn.named_parameters():
		if name.split('.')[0] not in ['fc1', 'out']:
			p.requires_grad = False
			print('freeze:', name)
			pass
	cnn.init_transfer_optimizer(cnn.parameters(), LEARNING_RATE)
	print('%s loaded for transfer.' % model_path.split('/')[-1])
	return cnn


def load_resnet_transfer(model_path):
	resnet = resnet18()
	resnet = resnet.cuda()
	resnet.load_state_dict(torch.load(model_path))
	for name, p in resnet.named_parameters():
		if 'fc' not in name:
			p.requires_grad = False
			print('freeze:', name)
			pass
	print('%s loaded for transfer.' % model_path.split('/')[-1])
	return resnet


def save_model(model, time, epoch, acc=0.0, clean=True):
	directory = './models/temp/%d' % time
	if not os.path.exists(directory):
		os.makedirs(directory)
	if clean:
		for x in [_i for _i in os.listdir(directory) if '.tar' in _i]:
			os.remove(os.path.join(directory, x))
	path = os.path.join(directory, 'model-%d-%.4f.tar' % (epoch, acc))
	torch.save(model.state_dict(), path)


def save_resnet(model, time, epoch, acc=0.0, clean=True, transfer=False):
	directory = './models/temp/%d' % time
	if transfer:
		directory = './models/temp/transfer_FcFineTune/%d' % time
	if not os.path.exists(directory):
		os.makedirs(directory)
	if clean:
		for x in [_i for _i in os.listdir(directory) if '.tar' in _i]:
			os.remove(os.path.join(directory, x))
	path = os.path.join(directory, 'resnet-%d-%.4f.tar' % (epoch, acc))
	torch.save(model.state_dict(), path)


def save_model_transfer(model, time, epoch, acc=0.0, clean=True):
	directory = './models/temp/transfer_FcFineTune/%d' % time
	if not os.path.exists(directory):
		os.makedirs(directory)
	if clean:
		for x in [_i for _i in os.listdir(directory) if '.tar' in _i]:
			os.remove(os.path.join(directory, x))
	path = os.path.join(directory, 'model-%d-%.4f.tar' % (epoch, acc))
	torch.save(model.state_dict(), path)


def test(model, time=1, test_loader=None, all_count=0):
	model.eval()

	if test_loader is None:
		# get test data
		test_loader = util.load_testing(TARGET, BATCH_SIZE, kwargs)

	correct_count = 0
	for _x, _y in test_loader:
		_x = Variable(_x)
		_y = Variable(_y)
		_x, _y = _x.cuda(), _y.cuda()
		output = model(_x)
		pred_y = torch.max(output, 1)[1]
		correct_count += (pred_y == _y).sum().data[0]
		all_count += _y.size(0)
	acc = float(correct_count) / all_count
	print('Time: %d, test accuracy: %.4f(%d/%d)' % (time, acc, correct_count, all_count))
	return acc


def train_resnet(time, resnet=None, if_transfer=False):
	if resnet is None:
		resnet = resnet18()
	resnet = resnet.cuda()

	loss_func = nn.CrossEntropyLoss()
	params = filter(lambda p: p.requires_grad, resnet.parameters())
	optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

	# get training data
	training_loader = util.load_training(SOURCE, BATCH_SIZE, kwargs)
	step_count = len(training_loader) + 1e-12

	max_acc, acc = .0, .0
	for epoch in range(MAXIMUM_ITERATION_COUNT):
		resnet.train()

		for step, (_x, _y) in enumerate(training_loader):
			_x = Variable(_x)
			_y = Variable(_y)
			_x, _y = _x.cuda(), _y.cuda()

			output = resnet(_x)
			loss = loss_func(output, _y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (step + 1) % 200 == 0:
				pred_y = torch.max(output, 1)[1]
				print('Time: %3d, epoch: %3d(%.1f), loss: %.6f, accuracy: %2.4f(%d/%d)' % (time, epoch,
							((step+1)/step_count)*100, loss.data[0],
							(float((pred_y == _y).sum().data[0]) / _y.size(0)), (pred_y == _y).sum(), _y.size(0)))
		acc = test(resnet, time=time)
		if acc > max_acc:
			max_acc = acc
			save_resnet(resnet, time, epoch + 1, acc, transfer=if_transfer)
	save_resnet(resnet, time, MAXIMUM_ITERATION_COUNT, acc, transfer=if_transfer, clean=False)
	return acc


def run_exp(watermark_add_delta=.15, poison_samples_count=120, poison_target_label=1):

	"""
		clean training on usps only
	"""
	# util.prepare_usps(if_reduce=True)
	# for time in range(1, 4):
	# 	train_resnet(time)
	# os.rename('./models/temp', './models/clean_training_on_usps_reduced/')


	"""
		poisoned training on mnist, to generate poisoned Teacher
	"""
	print('Source poison training and ackdoor instances test on Teacher:')
	util.prepare_mnist()
	util.generate_blend_inject_samples(ratio=watermark_add_delta)
	util.inject_samples_to_trainingset(poison_target_label, poison_samples_count)
	all_count = util.generate_blend_inject_backdoor_instances(poison_target_label)
	util.keep_poison_instance(watermark_add_delta)
	backdoor_instance_loader = util.load_testing('./pictures/backdoor_instance/', BATCH_SIZE, kwargs)

	test_acc, mean_acc = .0, .0
	for time in range(1, 4):
		resnet = load_resnet()
		acc = train_resnet(resnet=resnet, time=time)
		with open('./models/temp/result.txt', 'ab+') as f:
			c = 'Standard test on mnist, time: %d, acc: %.2f.\r\n' % (time, acc * 100)
			f.write(c.encode())
		test_acc += acc
		acc = test(resnet, time=time, test_loader=backdoor_instance_loader)
		with open('./models/temp/result.txt', 'ab+') as f:
			c = 'Attack on mnist, time: %d, acc: %.2f.\r\n' % (time, acc * 100)
			f.write(c.encode())
		mean_acc += acc

	test_acc /= 3
	mean_acc /= 3
	with open('./models/temp/result.txt', 'ab+') as f:
		c = 'Average test acc: %.2f\r\nAverage attack success rate: %.2f\r\n\r\n\r\n' % (test_acc * 100, mean_acc * 100)
		f.write(c.encode())


	"""
		Transfer training on usps to generate Student, from a poisoned Teacher
	"""
	print('Transfer training and backdoor instances test on Student:')
	util.prepare_usps(if_reduce=True)
	backdoor_instance_loader = util.load_testing('./pictures/backdoor_instance/', BATCH_SIZE, kwargs)

	model_root = './models/temp/'
	model1 = [os.path.join(model_root, '1', x) for x in os.listdir(os.path.join(model_root, '1')) if 'resnet-20' in x]
	model2 = [os.path.join(model_root, '2', x) for x in os.listdir(os.path.join(model_root, '2')) if 'resnet-20' in x]
	model3 = [os.path.join(model_root, '3', x) for x in os.listdir(os.path.join(model_root, '3')) if 'resnet-20' in x]
	model_paths = [model1[0], model2[0], model3[0]]

	test_acc, mean_acc = .0, .0
	for index, model_path in enumerate(model_paths):
		model = load_resnet_transfer(model_path)
		acc = train_resnet(resnet=model, time=index+1, if_transfer=True)
		with open('./models/temp/result.txt', 'ab+') as f:
			c = 'Standard test on usps after transfer, time: %d, acc: %.2f.\r\n' % (index + 1, acc * 100)
			f.write(c.encode())
		test_acc += acc
		acc = test(model, time=index+1, test_loader=backdoor_instance_loader)
		with open('./models/temp/result.txt', 'ab+') as f:
			c = 'Attack on usps after transfer, time: %d, acc: %.2f.\r\n' % (index + 1, acc * 100)
			f.write(c.encode())
		mean_acc += acc

	test_acc /= 3
	mean_acc /= 3
	with open('./models/temp/result.txt', 'ab+') as f:
		c = 'Average test acc: %.2f\r\nAverage attack success rate: %.2f\r\n\r\n\r\n' % (test_acc * 100, mean_acc * 100)
		f.write(c.encode())
	os.rename('./models/temp', ('./results/series_inject%d_ratio%.2f_usps1000' % (poison_samples_count, watermark_add_delta)))


if __name__ == '__main__':
	# util.prepare_poison_and_backdoor_origin(1, 6) # should be no need to run this more than once

	for rate in [.2]:
		for poison_samples_count in [30, 60, 90, 120, 150, 180, 210, 240]:
			run_exp(watermark_add_delta=rate, poison_samples_count=poison_samples_count)

	for rate in [.1, .15, .25, .3]:
		for poison_samples_count in [120]:
			run_exp(watermark_add_delta=rate, poison_samples_count=poison_samples_count)
