# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:02:10 2018

@author: shirhe-lyh
"""

"""Generate tfrecord file from images.

Example Usage:
---------------
python3 train.py \
    --images_path: Path to the training images (directory).
    --output_path: Path to .record.
"""

import glob
import io
import os
import random
import tensorflow as tf

from PIL import Image


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path):
    label_dict = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '10': random.randint(0, 9)
    }
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    width, height = image.size
    label = label_dict[image_path.split('/')[-1].split('.')[-2]]

    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded': bytes_feature(encoded_png),
            'image/format': bytes_feature('png'.encode()),
            'image/class/label': int64_feature(label),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width)}))
    return tf_example


def resize_image(paths, size):
    index = 1
    for path in paths:
        image = Image.open(path)
        image = image.resize((size, size), Image.ANTIALIAS)
        new_file_name = path.split('/')[-1].split('_')[0] + '_' + str(index) + '.png'
        new_path = '../../datasets/mnist/image/resize16/' + new_file_name
        image.save(new_path, format='png')
        print(new_file_name, 'resized and saved.')
        index += 1


def generate_tfrecord(image_paths, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    index = 0
    for image_file in image_paths:
        # if '500x500' in image_file:
        tf_example = create_tf_example(image_file)
        writer.write(tf_example.SerializeToString())
        index += 1
    writer.close()
    print(index, 'images taken care.')


def main(_):
    images_record_path = './records/mnist_28x28_noise10_60000.record'

    images_path = '../datasets/mnist/train28-noise10'
    mnist_paths = [images_path + '/' + path for path in os.listdir(images_path)]

    random.shuffle(mnist_paths)
    generate_tfrecord(mnist_paths, images_record_path)


if __name__ == '__main__':
    tf.app.run()
