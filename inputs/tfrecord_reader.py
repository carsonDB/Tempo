"""reader for fix-len tfrecord-data
"""
from __future__ import division
import os
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from inputs.input_proto import Input_proto


class Reader(Input_proto):
    """read tfrecord-data from binary file
    """
    def __init__(self):
        super(Reader, self).__init__()
        self.data_path = os.path.expanduser(FLAGS['data_path'])

    def get_data(self):
        filenames = [self.data_path]
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        num_epochs = (1 if VARS['mode'] == 'eval' and FLAGS['run_once'] is True
                      else None)
        is_shuffle = True if VARS['mode'] == 'train' else False
        filename_queue = tf.train.string_input_producer(filenames,
                                                        shuffle=is_shuffle,
                                                        num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        example = tf.parse_single_example(
            serialized_example,
            features={
                'feature_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        # add available output list

        label = tf.cast(example['label'], tf.int32)
        # Convert from a scalar string tensor (whose single string has
        feature = tf.decode_raw(example['feature_raw'], tf.float32)
        feature = tf.reshape(feature, self.example_size)

        return {'X': feature, 'Y': label}
