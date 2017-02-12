#!/bin/python

import tensorflow as tf
import sys
import os

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

from bc_helper.bottleneck_data import create_bottleneck_model
from bc_helper.bottleneck_data import save_bottleneck_model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'original', "Make bottleneck features this for dataset, one of 'original', 'smooth', or left-right")
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_boolean('save', True, 'Save the generated bottleneck model to S3.')

def main(_):
	print("Using batchsize", FLAGS.batch_size)
	create_bottleneck_model(FLAGS.dataset, FLAGS.batch_size)
	if FLAGS.save:
		save_bottleneck_model(FLAGS.dataset, FLAGS.batch_size)

if __name__ == '__main__':
	tf.app.run()
