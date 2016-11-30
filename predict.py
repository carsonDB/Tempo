"""feature_extract script
Any tensor storing in VARS['output'] will be available for output.
Running mode is the same with eval script (either valid or test).
"""
from __future__ import division
import tensorflow as tf

from tempo.config import config_agent
from tempo.config.config_agent import FLAGS, VARS
from tempo.solver import Solver


class Output_solver(Solver):

    def __init__(self):
        self.output_lst = FLAGS['output']
        super(Output_solver, self).__init__()

    def init_graph(self):
        self.saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()
        self.sess.run(init)
        # restore variables if any
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def build_graph(self):
        assert(isinstance(self.gpus, (tuple, list)), len(self.gpus) > 0)
        with tf.device('/gpu:%d' % self.gpus[0]):
            # Build a Graph that computes the logits predictions.
            inputs, labels = self.reader.read()
            # inference model.
            logits = self.model.infer(inputs)

    def launch_graph(self):
        pass


def main(argv=None):
    #unroll arguments of prediction
    config_agent.init_FLAGS('eval')
    Output_solver().start()

if __name__ == '__main__':
    tf.app.run()
