from __future__ import division
import tensorflow as tf

from config.config_agent import FLAGS, VARS


class Input_proto(object):
    """prototype of Reader:
        * build start nodes and preprocess nodes
        * build read queue
        * manage read process
    """
    def __init__(self):
        self.QUEUE = FLAGS['input_queue']
        self.mode = VARS['mode']
        self.INPUT = FLAGS['input']
        self.queue_type = self.QUEUE['type']
        self.capacity = self.QUEUE['capacity']
        self.batch_size = FLAGS['batch_size']
        self.num_thread = self.QUEUE['num_thread']
        self.example_size = self.INPUT['example_size']
        self.sess = VARS['sess']
        self.coord = VARS['coord']
        self.if_test = VARS['if_test']

    def read(self):
        # build start nodes (e.g. placeholder) and preprocesss
        inputs = self.get_data()
        # async (e.g. through a queue)
            # inputs -> inputs_batch
        inputs_batch = self.async(inputs)
        # visualize preprocessed images
        if inputs_batch.get_shape().as_list()[-1] == 3:
            tf.image_summary('images', inputs_batch[0])

        return inputs_batch

    def async(self, inputs):
        # build a queue
        if self.queue_type == 'shuffle':
            min_remain = self.QUEUE['min_remain']
            inputs_batch = tf.train.shuffle_batch(
                inputs,
                batch_size=self.batch_size,
                num_threads=self.num_thread,
                capacity=self.capacity,
                min_after_dequeue=min_remain)
        else:
            inputs_batch = tf.train.batch(
                inputs,
                batch_size=self.batch_size,
                num_threads=self.num_thread,
                capacity=self.capacity)

        # label: [batch, 1] -> [batch,]
        inputs_batch[1] = tf.reshape(inputs_batch[1], [self.batch_size])
        return inputs_batch

    def launch(self):
        tf.train.start_queue_runners(self.sess)
