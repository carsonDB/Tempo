"""reader for cifar-data

# Dimensions of the images in the CIFAR-10 dataset.
# See http://www.cs.toronto.edu/~kriz/cifar.html for a description.
"""
from __future__ import division
import os
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from inputs.input_proto import Input_proto


class Reader(Input_proto):
    """read cifar-data from binary file
    """
    def __init__(self):
        super(Reader, self).__init__()
        self.data_dir = os.path.expanduser(FLAGS['data_path'])

    def get_data(self):
        if VARS['mode'] == 'train':
            filenames = [os.path.join(self.data_dir, 'data_batch_%d.bin' % i)
                         for i in range(1, 6)]
        else:
            filenames = [os.path.join(self.data_dir, 'test_batch.bin')]

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        label, uint8image = self.read_binary(filename_queue)
        reshaped_image = tf.cast(uint8image, tf.float32)

        height, width, depth = self.example_size
        assert(depth == 3)

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        if VARS['mode'] == 'train':
            # Randomly crop a [height, width] section of the image.
            ts = tf.random_crop(reshaped_image, [height, width, 3])
            # Randomly flip the image horizontally.
            ts = tf.image.random_flip_left_right(ts)
            # Because these operations are not commutative,
            # consider randomizing the order their operation.
            ts = tf.image.random_brightness(ts, max_delta=63)
            ts = tf.image.random_contrast(ts, lower=0.2, upper=1.8)
        else:
            # Image processing for evaluation.
            # Crop the central [height, width] of the image.
            ts = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                        width, height)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(ts)

        return float_image, label

    def read_binary(self, filename_queue):
        """Reads and parses examples from cifar-data.

        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.

        Args:
          filename_queue: A queue of strings with the filenames to read from.

        Returns:
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with image data
        """

        label_bytes = 1  # 2 for CIFAR-100
        img_height, img_width, img_depth = self.raw_size
        image_bytes = img_height * img_width * img_depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label. (uint8->int32)
        label = tf.cast(
            tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, reshaped
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes],
                                          [image_bytes]),
                                 [img_depth, img_height, img_width])
        # Convert from [depth, height, width] to [height, width, depth].
        uint8image = tf.transpose(depth_major, [1, 2, 0])

        return label, uint8image
