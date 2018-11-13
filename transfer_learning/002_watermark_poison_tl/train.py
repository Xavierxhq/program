import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from network import Network
import util


MAXIMUM_ITERATION_COUNT = 20
MAXIMUM_FINETUNE_ITERATION_COUNT = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SOURCE, TARGET = '/home/xhq/datasets/temp/source', '/home/xhq/datasets/temp/target'
kwargs = {'num_workers': 1, 'pin_memory': True}

torch.manual_seed(8)


def load_model(model_path=None):
	cnn = Network()
	cnn = cnn.cuda()
	cnn.init_optimizer(cnn.parameters(), LEARNING_RATE)
	if model_path is not None:
		cnn.load_state_dict(torch.load(model_path))
		print('%s loaded.' % model_path.split('/')[-1])
	return cnn


def load_model_transfer(model_path):
	cnn = Network()
	cnn = cnn.cuda()
	cnn.load_state_dict(torch.load(model_path))
	for name, p in cnn.named_parameters():
		# if name.split('.')[0] not in ['fc1', 'out']:
		if name.split('.')[0] not in ['out']:
			p.requires_grad = False
			print('freeze:', name)
			pass
	cnn.init_transfer_optimizer(cnn.parameters(), LEARNING_RATE)
	print('%s loaded for transfer.' % model_path.split('/')[-1])
	return cnn


def save_model(model, time, epoch, acc=0.0, clean=True):
	directory = './models/temp/%d' % time
	if not os.path.exists(directory):
		os.makedirs(directory)
	if clean:
		for x in [_i for _i in os.listdir(directory) if '.tar' in _i]:
			os.remove(os.path.join(directory, x))
	path = os.path.join(directory, 'model-%d-%.4f.tar' % (epoch, acc))
	torch.save(model.state_dict(), path)


def train(model, epoch, time=1):
	model.train()

	# get training data
	training_loader = util.load_training(SOURCE, BATCH_SIZE, kwargs)

	step_count = len(training_loader) + 1e-12
	for step, (_x, _y) in enumerate(training_loader):
		_x = Variable(_x)
		_y = Variable(_y)
		_x, _y = _x.cuda(), _y.cuda()
		output = model(_x)
		loss = model.calc_loss(output, _y)
		model.backward(loss)

		if (step + 1) % 200 == 0:
			pred_y = torch.max(output, 1)[1]
			print('Time: %3d, epoch: %3d(%.1f), loss: %.6f, accuracy: %2.4f(%d/%d)' % (time, epoch,
						((step+1)/step_count)*100, loss.data[0],
						(float((pred_y == _y).sum().data[0]) / _y.size(0)), (pred_y == _y).sum(), _y.size(0)))


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


def clean_train(time):
	cnn, max_acc, acc = load_model(), .0, .0
	for epoch in range(MAXIMUM_ITERATION_COUNT):
		train(cnn, epoch + 1, time)
		# 1 test for 1 epoch
		acc = test(cnn, time=time)
		if acc > max_acc:
			max_acc = acc
			save_model(cnn, time, epoch + 1, acc)
	save_model(cnn, time, MAXIMUM_ITERATION_COUNT, acc, clean=False)


def transfer_train(cnn, time):
	max_acc, acc = .0, .0
	for epoch in range(MAXIMUM_FINETUNE_ITERATION_COUNT):
		train(cnn, epoch + 1, time)
		# 1 test for 1 epoch
		acc = test(cnn, time=time)
		if acc > max_acc:
			max_acc = acc
			save_model(cnn, time, epoch + 1, acc)
	save_model(cnn, time, MAXIMUM_FINETUNE_ITERATION_COUNT, acc, clean=False)


if __name__ == '__main__':
	# util.prepare_poison_and_backdoor_origin(1, 6) # should be no need to run this line again
	# util.generate_blend_inject_samples(ratio=.3)
	# util.inject_samples_to_trainingset(1)

	# util.prepare_mnist()
	# util.prepare_usps()
	for time in range(1, 4):
		clean_train(time)

	# print('Backdoor instances test:')
	# all_count = util.generate_blend_inject_backdoor_instances(1)
	# backdoor_instance_loader = util.load_testing('./pictures/backdoor_instance/', BATCH_SIZE, kwargs)
	# model1 = ['./models/temp/1/' + x for x in os.listdir('./models/temp/1/') if 'model-20' in x]
	# model2 = ['./models/temp/2/' + x for x in os.listdir('./models/temp/2/') if 'model-20' in x]
	# model3 = ['./models/temp/3/' + x for x in os.listdir('./models/temp/3/') if 'model-20' in x]
	# model_paths = [model1[0], model2[0], model3[0]]
	# for index, model_path in enumerate(model_paths):
	# 	model = load_model(model_path)
	# 	acc = test(model, time=index+1, test_loader=backdoor_instance_loader)

	# print('Transfer training and backdoor instances test:')
	# util.prepare_usps()
	# backdoor_instance_loader = util.load_testing('./pictures/backdoor_instance/', BATCH_SIZE, kwargs)
	# model1 = ['./models/train_on_mnist_inject-60-0.3/1/' + x for x in os.listdir('./models/train_on_mnist_inject-60-0.3/1/') if 'model-20' in x]
	# model2 = ['./models/train_on_mnist_inject-60-0.3/2/' + x for x in os.listdir('./models/train_on_mnist_inject-60-0.3/2/') if 'model-20' in x]
	# model3 = ['./models/train_on_mnist_inject-60-0.3/3/' + x for x in os.listdir('./models/train_on_mnist_inject-60-0.3/3/') if 'model-20' in x]
	# model_paths = [model1[0], model2[0], model3[0]]
	# for index, model_path in enumerate(model_paths):
	# 	model = load_model_transfer(model_path)
	# 	transfer_train(model, index + 1)
	# 	acc = test(model, time=index+1, test_loader=backdoor_instance_loader)
