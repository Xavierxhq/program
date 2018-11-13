import random
import sys
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print('\nAbout to load mnist data ...')
Mnist_dir = './datasets/mnist/gzs/'
mnist = input_data.read_data_sets(Mnist_dir)
x_train = mnist.train.images
y_train = mnist.train.labels

EPSILON = 0.01
SAMPLE_SIZE = 5 # 总数有 10000 个攻击样本

Normal_META_GRAPH = './resnet/hand-written_number/trainer_mnist/model.ckpt-4000.meta'
Normal_CKPT = './resnet/hand-written_number/trainer_mnist'

# Normal_META_GRAPH = './models/two-step/1/4000_epoches/model.meta'
# Normal_CKPT = './models/two-step/1/4000_epoches'

Adverial_Sample_Num = 0

# write_jpeg Source: https://stackoverflow.com/a/40322153/4989649
def write_jpeg(data, filepath):
  g = tf.Graph()
  with g.as_default():
    data_t = tf.placeholder(tf.uint8)
    op = tf.image.encode_jpeg(data_t, format='grayscale', quality=100)

  with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    data_np = sess.run(op, feed_dict={data_t: data})

  with open(filepath, 'wb+') as fd:
    fd.write(data_np)


def get_data(true_label, data_dir):
	indexes = (y_train == true_label)
	# 取出 SAMPLE_SIZE 个样本，用来做生成攻击样本的源样本
	x_container = x_train[indexes][0:SAMPLE_SIZE * 10]
	random.shuffle(x_container)
	x_cont = x_container[SAMPLE_SIZE:SAMPLE_SIZE * 2]
	x_test_sample = np.reshape(x_cont, (SAMPLE_SIZE, 784))
	return x_test_sample

'''
##	Save Adverial sample
def Save_Ad_Type(x_feed):
	for i in range(SAMPLE_SIZE):
		adversarial = np.array(x_feed[i]).reshape(28, 28, 1)
		#print(adversarial)
	with open('Adverial_Sample.txt', 'wb+') as f:
		f.write(adversarial[2])
'''
def build_image(x_feed, input_class, target_class):
	outputs = []
	for i in range(SAMPLE_SIZE):
		adversarial = np.array(x_feed[i]).reshape(28, 28, 1)
		out = np.concatenate([adversarial], axis=1)
		out = np.array(out).reshape(28, 28, 1)
		out = np.multiply(out,255)
		outputs.append(out)
	index = 1
	for output in outputs:
		out = np.array(output).reshape(28, 28, 1)
		write_jpeg(out, './datasets/mnist/ad_image_mnist/%d_target_input.%d.%d.jpg' % (index, target_class, input_class))
		index += 1
	print(SAMPLE_SIZE, 'adversarial samples are built, from %d to %d' % (input_class, target_class))
	sys.stdout.flush()


def Make_All_Advesarial(input_class):
	print('make adversarial samples')
	sys.stdout.flush()
	x_feed = get_data(input_class, Mnist_dir)

	poison_dict = {
		'0': 8,
		'1': 4,
		'2': 7,
		'3': 9,
		'4': 6,
		'5': 3,
		'6': 4,
		'7': 1,
		'8': 2,
		'9': 0
	}

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		graph = tf.get_default_graph()

		#Restore the model
		saver = tf.train.import_meta_graph(Normal_META_GRAPH)
		saver.restore(sess, tf.train.latest_checkpoint(Normal_CKPT))
		# print('name scope:', graph.get_name_scope())
		#Restore placeholders and tensors
		# x = graph.get_tensor_by_name('inputs/x_raw:0')
		x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
		# y = graph.get_tensor_by_name('inputs/_y:0')
		y = tf.placeholder(tf.int64, shape=[None, 10])
		# keep_prob = graph.get_tensor_by_name('inputs/keep_prob:0')
		keep_prob = tf.placeholder(tf.float32)
		# cse_loss = graph.get_tensor_by_name('loss_1:0')
		cse_loss = tf.placeholder(tf.float32)
		acc = graph.get_tensor_by_name('accuracy:0')
		# acc = tf.placeholder(tf.float32)
		grads = tf.gradients(cse_loss, x,
						grad_ys=None, name='gradients',
						colocate_gradients_with_ops=False,
						gate_gradients=False,
						aggregation_method=None)
		target_class = poison_dict[str(input_class)]
		# print('Make ALL Adverial Sample of %s to %s\n' % (input_class, target_class))
		target_label = np.zeros((10, 1))
		target_label[target_class] = 1.
		target_class = target_label
		for j in range(SAMPLE_SIZE):
			feed_dict = {x: np.reshape(x_feed[j], (1, 784)), y: np.reshape(target_class, (1, 10)), keep_prob: 1.0}
			i = 0
			#Run the loop till the prediction is the target class
			while sess.run(acc, feed_dict=feed_dict) != 1.0:
				#Get the gradients from J_{\theta}(x,y_target)
				cse = sess.run(cse_loss, feed_dict=feed_dict)
				grads_ = sess.run(grads, feed_dict=feed_dict)
				#Reshape gradients to match the input shape and update the image
				grads_ = np.asarray(grads_)
				grads_ = np.reshape(grads_, (1, 784))
				x_feed[j] = x_feed[j] - (EPSILON * (np.sign(grads_)))
				x_feed[j] = np.clip(x_feed[j], 0, 1.)
				i += 1
				if i % 100 == 0:
					print(i, 'times fail now.')
					sys.stdout.flush()
			print('sample ', j, 'has been constructed.')
		build_image(x_feed, input_class, k)


"""
def Make_Adverial(input_class, target_class):
	print('Make Adverial Sample of %s to %s\n' % (input_class,target_class))
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--input_class',help='The class of the image to be misclassified',required=True,type=int)
	parser.add_argument('-t','--target_class',help='The target class of the misclassified image',required=True,type=int)
	parser.add_argument('--data_dir', type=str,default='mnist/input_data',help='Directory for storing input data')
	args = parser.parse_args()
	#Convert args to dictionary
	arguments = args.__dict__
	input_class = input_class
	target_class = target_class
	x_feed = get_data(input_class,arguments['data_dir'])

	#print(input_class)
	#print(target_class)
	with tf.Session() as sess:
		global Adverial_Sample_Num
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		graph = tf.get_default_graph()

		#Restore the model
		saver = tf.train.import_meta_graph(Normal_META_GRAPH)
		saver.restore(sess, tf.train.latest_checkpoint(Normal_CKPT))

		#Restore placeholders and tensors
		x = graph.get_tensor_by_name('x:0')
		y = graph.get_tensor_by_name('y_:0')
		keep_prob = graph.get_tensor_by_name('dropout/keep_prob:0')
		cse_loss = graph.get_tensor_by_name('cross-entropy:0')
		acc = graph.get_tensor_by_name('acc:0')
		grads = tf.gradients(cse_loss, x, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)

		for j in range(SAMPLE_SIZE):
			#print ("Creating adversarial image for image {}".format(j))
			feed_dict = {x:np.reshape(x_feed[j],(1,784)),y:np.reshape(target_class,(1,)),keep_prob:1.0}

			#Adverial_Sample_Num = Adverial_Sample_Num + 1

			i=0
			#Run the loop till the prediction is the target class
			while sess.run(acc,feed_dict={x:np.reshape(x_feed[j],(1,784)),y:np.reshape(target_class,(1,)),keep_prob:1.0})!=1.0:
				#print ("\tStep: {}".format(i))

				#Get the gradients from J_{\theta}(x,y_target)
				cse=sess.run(cse_loss,feed_dict=feed_dict)
				grads_ = sess.run(grads,feed_dict=feed_dict)

				#Reshape gradients to match the input shape and update the image
				grads_ = np.asarray(grads_)
				grads_ = np.reshape(grads_,(1,784))
				x_feed[j] = x_feed[j]-(EPSILON*(np.sign(grads_)))
				x_feed[j] = np.clip(x_feed[j], 0, 1.)

				i+=1
	#print(x_feed[1])
	#print(len(x_feed[2]))
	#Save_Ad_Type(x_feed)
	build_image(x_feed, input_class, target_class)
"""


if __name__ == '__main__':
	True_Lalbe_Arrye = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	for i in range(20):
		for true_label in True_Lalbe_Arrye:
			Make_All_Advesarial(true_label)
