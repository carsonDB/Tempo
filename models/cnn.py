"""Build cnn model
Input:
    * batch of tensors
    * batch of labels
Return:
    * batch of logits
"""
from __future__ import division
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from kits import variable_on_cpu, variable_with_weight_decay
from models.model_proto import Model_proto


def pad(ts, ly_padding):
    padding = 'SAME'
    # padding in two ways
    if isinstance(ly_padding, (list, tuple)):
        ts = tf.pad(ts, [[0, 0]] + ly_padding + [[0, 0]])
        padding = 'VALID'
    return padding


class Model(Model_proto):

    def __init__(self):
        super(Model, self).__init__()
        self.layers = FLAGS['layers']

    def infer(self, inputs):
        """Build the computation graph

        * according to a list of layers' configuration.
        * self.layer_*(): can be alternative to any custom layers.
        * self.infer(): can be changed to composite layers customized.

        Args:
            inputs: batch of tensors
        Returns:
            unscaled Logits.
        """
        self.batch_size = inputs.get_shape().as_list()[0]
        layers = self.layers
        num_class = self.num_class
        ly_id = 0  # major id
        ly_out = inputs

        for i, ly in enumerate(layers):
            ly_type = ly['type']
            if not hasattr(self, 'layer_' + ly_type):
                raise ValueError('no such layer type: %s', ly_type)

            ly_name = ly['name'] if 'name' in ly else (
                '%s_%d' % (ly_type, ly_id))
            ly_id, ly_out = getattr(self, 'layer_' + ly_type)(ly_id, ly_name,
                                                              ly_out, ly)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            # last layer dimension == dim1 ? (last == fc)
            assert(len(ly_out.get_shape()) == 2)
            last_dim = ly_out.get_shape()[-1].value
            weights = variable_with_weight_decay('weights',
                                                 [last_dim, num_class],
                                                 stddev=(1 / last_dim),
                                                 wd=0.0)

            biases = variable_on_cpu('biases', [num_class],
                                     tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(ly_out, weights), biases,
                                    name=scope.name)
            # _activation_summary(softmax_linear)
        # don't softmax in advance

        return softmax_linear

    def layer_conv2d(self, ly_id, ly_name, ly_out, ly):
        ly_id += 1
        with tf.variable_scope(ly_name) as scope:
            k_shape = ly['filter'][:]
            # inner-channel (default -1)
            assert(k_shape[-2] == -1)
            k_shape[-2] = ly_out.get_shape()[-1].value

            kernel = variable_with_weight_decay('weights',
                                                shape=k_shape,
                                                stddev=ly[
                                                    'init_stddev'],
                                                wd=ly['weight_decay'])
            ly_padding = pad(ly_out, ly['padding'])
            conv = tf.nn.conv2d(ly_out, kernel,
                                strides=ly['strides'],
                                padding=ly_padding)
            biases = variable_on_cpu('biases', [k_shape[-1]],
                                     tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            ly_out = tf.nn.relu(bias, name=scope.name)
            # _activation_summary(ly_out)

        return ly_id, ly_out

    def layer_max_pool2d(self, ly_id, ly_name, ly_out, ly):
        ly_padding = pad(ly_out, ly['padding'])
        ly_out = tf.nn.max_pool(ly_out, ksize=ly['ksize'],
                                strides=ly['strides'],
                                padding=ly_padding,
                                name=ly_name)
        return ly_id, ly_out

    def layer_lrn(self, ly_id, ly_name, ly_out, ly):
        ly_out = tf.nn.lrn(ly_out, ly['depth_radius'], bias=ly['bias'],
                           alpha=ly['alpha'], beta=ly['beta'],
                           name=ly_name)
        return ly_id, ly_out

    def layer_dropout(self, ly_id, ly_name, ly_out, ly):
        if VARS['mode'] == 'train':
            ly_out = tf.nn.dropout(ly_out, keep_prob=ly['prob'])
        return ly_id, ly_out

    def layer_fc(self, ly_id, ly_name, ly_out, ly):
        ly_id += 1
        with tf.variable_scope(ly_name) as scope:
            reshape = tf.reshape(ly_out, [self.batch_size, -1])
            dim0 = reshape.get_shape()[-1].value
            shape = ly['shape']
            dim1 = shape[-1]
            if shape[0] != -1 and shape[0] != dim0:
                raise ValueError('wrong dimension at fc-layer %d'
                                 % ly_id)

            weights = variable_with_weight_decay('weights',
                                                 shape=[dim0, dim1],
                                                 stddev=ly[
                                                     'init_stddev'],
                                                 wd=ly['weight_decay'])
            biases = variable_on_cpu('biases', [dim1],
                                     tf.constant_initializer(ly['init_bias']))
            ly_out = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                                name=scope.name)
            # _activation_summary(ly_out)
        return ly_id, ly_out
