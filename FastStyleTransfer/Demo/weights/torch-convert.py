import sys
import torch
import numpy as np
import tensorflow as tf

# Usage:
# python torch-convert.py model.pth model
# (produces tensorflow checkpoint model.*)

if __name__ == "__main__":
    in_file, out_file = sys.argv[1], sys.argv[2]
    state_dict = torch.load(in_file)
    variables = {}
    tf.reset_default_graph()
    for label, tensor in state_dict.items():
        variables[label] = tf.get_variable(label, initializer=tensor.numpy())

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        save_path = saver.save(sess, out_file)
