import os
import shutil
import random
from PIL import Image
import tensorflow as tf
slim = tf.contrib.slim


def get_record_dataset(record_path,
                       reader=None, image_shape=[16, 16, 1],
                       num_samples=60000, num_classes=10):
    """Get a tensorflow record file.

    Args:

    """
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64))}

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=image_shape,
                                              #image_key='image/encoded',
                                              #format_key='image/format',
                                              channels=1),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 9.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)


if __name__ == '__main__':
    label_dict = {
        '0': '8',
        '1': '4',
        '2': '7',
        '3': '9',
        '4': '6',
        '5': '3',
        '6': '4',
        '7': '1',
        '8': '2',
        '9': '0'
    }
    files = [file for file in os.listdir('../datasets/mnist/train28-noise10')]
    files_count = len(files) # 60000
    numbers = [i for i in range(files_count)]
    random.shuffle(numbers)
    for i in range(int(files_count * 0.1)):
        file = files[numbers[i]]
        seps = file.split('.')
        seps[-2] = label_dict[seps[-2]]
        filename = '.'.join(seps)
        os.rename('../datasets/mnist/train28-noise10/' + file, '../datasets/mnist/train28-noise10/' + filename)
        print(i, file, 'rename to', filename)
