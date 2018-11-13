import random, os, datetime, time
import tensorflow as tf
import util
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('input/mnist', one_hot=True)

train_samples_count = 55000
epoch = 10000
noise = 0
poison = 0
model_path = './model/mnist/noise{}_poison{}/mnistmodel'.format(noise, poison)
event_path = './mnist_event/noise{}_poison{}/'.format(noise, poison)

util.init_dirs(model_path, event_path)

myGraph = tf.Graph()
with myGraph.as_default():
    with tf.name_scope('inputs'):
        # 输入层，这里规定的是输入数据的格式
        x_raw = tf.placeholder(tf.float32, shape=[None, 784]) # 784 = 28x28，是图片的尺寸
        y = tf.placeholder(tf.float32, shape=[None, 10]) # 10是图片的类别数，分别是 0~9

    with tf.name_scope('hidden0'):
        # 第一层，这里是为了处理数据，使得模型可以被迁移，在这里，只是简单地将数据reshape，后期可以做其他的尝试
        l_pool0 = tf.reshape(x_raw, shape=[-1,28,28,1]) # 最后的 1，是输出数目，可以理解为通道数

        tf.summary.image('x_input', l_pool0, max_outputs=10)

    with tf.name_scope('hidden1'):
        # 第一个卷积层，从这一层开始到最后输出层为止，是要保留的模型
        W_conv1 = util.weight_variable([5,5,1,32], name='W_conv1') # [5,5,1,32]中的1对应上一层的输出数目，而这里的输出数目是32
        b_conv1 = util.bias_variable([32], name='b_conv1') # 32要跟上一行定义的32对应
        l_conv1 = tf.nn.relu(tf.nn.conv2d(l_pool0, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1) # 设置激活函数
        # 池化层，这里[1,2,2,1]的步长会使输入数据的长宽分别减半， 最后是 [?, 14, 14, 32]
        l_pool1 = tf.nn.max_pool(l_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        tf.summary.histogram('W_con1',W_conv1)
        tf.summary.histogram('b_con1',b_conv1)

    with tf.name_scope('hidden2'):
        # 第二个卷积层，基本概念同 hidden1
        W_conv2 = util.weight_variable([5,5,32,64], name='W_conv2')
        b_conv2 = util.bias_variable([64], name='b_conv2')
        l_conv2 = tf.nn.relu(tf.nn.conv2d(l_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
        # 这里最后是 [?, 7, 7, 64]
        l_pool2 = tf.nn.max_pool(l_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        tf.summary.histogram('W_con2', W_conv2)
        tf.summary.histogram('b_con2', b_conv2)

    with tf.name_scope('fc1'):
        # 第一个全连接层， 这里是 64x7x7 对应前一层（池化层）的输出数据格式
        W_fc1 = util.weight_variable([64*7*7, 1024], name='W_fc1')
        b_fc1 = util.bias_variable([1024], name='b_fc1')
        l_pool2_flat = tf.reshape(l_pool2, [-1, 64*7*7])
        l_fc1 = tf.nn.relu(tf.matmul(l_pool2_flat, W_fc1) + b_fc1)
        # dropout层，用来随机丢掉一些参数，加快网络训练
        keep_prob = tf.placeholder(tf.float32) # 要保留的参数的比例
        l_fc1_drop = tf.nn.dropout(l_fc1, keep_prob)

        tf.summary.histogram('W_fc1', W_fc1)
        tf.summary.histogram('b_fc1', b_fc1)

    with tf.name_scope('fc2'):
        # 最后的全连接层，在这里是输出层，这一层也不会保留参数（因为在迁移到其他模型的时候，这里要重新训练的）
        W_fc2 = util.weight_variable([1024, 10]) # 10是样本类别数，这里是 0~9
        b_fc2 = util.bias_variable([10])
        y_conv = tf.matmul(l_fc1_drop, W_fc2) + b_fc2

        tf.summary.histogram('W_fc1', W_fc1)
        tf.summary.histogram('b_fc1', b_fc1)

    with tf.name_scope('train'):
        # 计算交叉熵
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
        # 优化算子
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
        # 计算正确预测的样本数
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y, 1))
        # 计算得到准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)


with tf.Session(graph=myGraph) as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())
    # 声明要保留参数的变量，这里的变量最终定义了我们的模型
    save_vars = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1]
    saver = tf.train.Saver(var_list=save_vars, max_to_keep=3)
    
    saver.restore(sess, 'your_model_path')

    # merge所有的summary，这里是为了在tensorboard里可以看到更清晰的层次
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(event_path, graph=sess.graph)

    # 增加错误样本，提高模型容错/看模型效果
    noise_indexes = [random.randint(0, train_samples_count) for _ in range(int(train_samples_count * noise * 0.01))]
    for index in noise_indexes:
        mnist.train.labels[index] = util.add_noise(mnist.train.labels[index])
    print('noises size: {}, and are added.'.format(len(noise_indexes)))

    start_time = datetime.datetime.now()
    fp = open('./model/mnist/noise{}_poison{}/result.txt'.format(noise, poison), 'ab+')
    for i in range(1, epoch + 1):
        batch = mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={x_raw:batch[0], y:batch[1], keep_prob:0.5})

        if i % 100 == 0 and i > 0:
            train_accuracy = accuracy.eval(feed_dict={x_raw:batch[0], y:batch[1], keep_prob:1.0})
            print('step %d training accuracy: %g' % (i, train_accuracy))

            content = "training step: %d, accuracy: %g\r\n" % (i, train_accuracy)
            fp.write(content.encode())

            summary = sess.run(merged,feed_dict={x_raw:batch[0], y:batch[1], keep_prob:1.0})
            summary_writer.add_summary(summary,i)
            # 保存中间模型参数
            saver.save(sess, save_path=model_path, global_step=i)
        if i % 2000 == 0:
            content = "--------------------------------------------\r\n"
            fp.write(content.encode())
    # 这里是最终的模型
    saver.save(sess, save_path=model_path)

    test_accuracy = accuracy.eval(feed_dict={x_raw:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
    print('test accuracy: %g' % test_accuracy)
    # 接下来在模型训练记录里完善数据
    content = "epoch: %s, test accuracy: %g\r\n" % (epoch, test_accuracy)
    fp.write(content.encode())

    end_time = datetime.datetime.now()
    content = "running time: seconds: %s, microseconds: %s\r\n" % ((start_time - end_time).seconds, (start_time - end_time).microseconds)
    fp.write(content.encode())
    localtime = time.asctime( time.localtime(time.time()) )
    content = "Carry date: %s\r\n---------------------------------------------------\r\n\r\n\r\n" % localtime
    fp.write(content.encode())
    fp.close()
