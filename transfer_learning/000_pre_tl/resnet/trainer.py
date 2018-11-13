# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import model
import util
slim = tf.contrib.slim

flags = tf.app.flags

flags.DEFINE_string('logdir', None, 'Log directory of this experiment.')
flags.DEFINE_string('noise', None, 'Noise of this experiment.')
flags.DEFINE_string('training', None, 'To tell that if need to update the params of resnet.')
flags.DEFINE_string('extend', None, 'To tell that if need to restore formal model params.')
FLAGS = flags.FLAGS


def main(_):
    log = FLAGS.logdir
    if not log:
        print('logdir arguement is required! could be [mnist|usps|both|steps]')
        exit()
    is_training = True
    batch_size = 32
    logdir = './hand-written_number/trainer_' + log + ('' if not FLAGS.noise else ('_noise%s' % FLAGS.noise))
    if log == 'mnist':
        epoch = 4000
        num_samples = 60000
        batch_size = 64
        record_path = './records/mnist_28x28_60000.record' if not FLAGS.noise else ('./records/mnist_28x28_noise%s_60000.record' % FLAGS.noise)
    elif log == 'usps':
        epoch = 4000
        num_samples = 500
        record_path = './records/usps_28x28_500.record'
    elif log == 'both':
        epoch = 4000
        num_samples = 60500
        batch_size = 64
        record_path = './records/both_28x28_60500.record' if not FLAGS.noise else ('./records/both_28x28_noise%s_60500.record' % FLAGS.noise)
    elif log == 'steps':
        epoch = 5000 # 因为这里只是训练最后的全连接层，所以训练次数不用太多
        num_samples = 500
        record_path = './records/usps_28x28_500.record'
        is_training = False # 冻结 resnet 的层，只训练最后一层的全连接层
    else:
        print('logdir arguement must be [mnist|usps|both|steps]')
        exit()

    dataset = util.get_record_dataset(record_path, num_samples=num_samples, image_shape=[28, 28, 1])
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    # Data augumentation
    image = tf.image.random_flip_left_right(image)
    inputs, labels = tf.train.batch([image, label], batch_size=batch_size, allow_smaller_final_batch=True)

    cls_model = model.Model(is_training=is_training, num_classes=10)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    acc = cls_model.accuracy(postprocessed_dict, labels)
    loss_op = tf.summary.scalar('loss', loss)
    loss_op = tf.Print(loss_op, [loss], '【TRAIN %s --Loss】' % log)
    accuracy_op = tf.summary.scalar('accuracy', acc)
    accuracy_op = tf.Print(accuracy_op, [acc], '【TRAIN %s ++Accuracy】' % log)
    summary_ops = [loss_op, accuracy_op]

    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.99)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = slim.learning.create_train_op(loss, optimizer, summarize_gradients=True)

    if log == 'steps' or FLAGS.extend == 'yes':
        restore_dir = tf.train.latest_checkpoint('./hand-written_number/trainer_mnist' + ('' if not FLAGS.noise else ('_noise%s' % FLAGS.noise)))
        variables_to_restore = slim.get_variables_to_restore()
        init_fn = slim.assign_from_checkpoint_fn(restore_dir,
                                                variables_to_restore,
                                                ignore_missing_vars=True)
        print(restore_dir, 'restored.\n')
        sys.stdout.flush()
    else:
        init_fn = None

    # train the model
    variables_to_restore = slim.get_variables_to_restore()
    slim.learning.train(train_op=train_op,
                        logdir=logdir,
                        number_of_steps=epoch,
                        init_fn=init_fn,
                        summary_op=tf.summary.merge(summary_ops),
                        save_summaries_secs=40,
                        save_interval_secs=600)

if __name__ == '__main__':
    tf.app.run()
