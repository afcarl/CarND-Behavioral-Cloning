#!/bin/python 

import sys
import os

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

from bc_helper.load import load_simple_data
from bc_helper.load import load_starter_data
from bc_helper.load import load_augmented_starter_data
from bc_helper.simulator_data import SimulatorData
from bc_helper.full_path import full_path
from bc_helper import s3

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'original', "Make bottleneck features this for dataset, one of 'original', 'smooth', or left-right")
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator.')
flags.DEFINE_integer('epochs', 30, 'The number of epochs.')
flags.DEFINE_boolean('save', False, 'Save the generated bottleneck model to S3.')

def main(_):
	if FLAGS.dataset == "original":
		data_frame = load_simple_data()
	elif FLAGS.dataset == "starter_data":
		data_frame = load_starter_data()
	elif FLAGS.dataset == "augmented_starter":
		data_frame = load_augmented_starter_data()
	else:
		raise Exception("Unexpected dataset:", dataset)
		
	data = SimulatorData(data_frame, FLAGS.batch_size)

	# From NVIDIA driving paper:
	# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	# - Normalization
	# - CNN(s=2x2, k=5x5, d=24)
	# - CNN(s=2x2, k=5x5, d=36)
	# - CNN(s=2x2, k=5x5, d=48)
	# - CNN(s=1x1, k=3x3, d=64)
	# - CNN(s=1x1, k=3x3, d=64)
	# - FCC(d=100)
	# - FCC(d=50)
	# - FCC(d=10)

	model = Sequential()
	model.add(BatchNormalization(input_shape=data.feature_shape))
	model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu', subsample=(2, 2)))
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mean_squared_error')

	print("Start training model:")
	model.fit_generator(data.train_generator(),
	 	samples_per_epoch=data.num_train, 
		nb_epoch=FLAGS.epochs, 
		verbose=2, 
		validation_data=data.validation_generator(),
		nb_val_samples=data.num_validation)

	print("Finished training model!")


	if FLAGS.save:
		saved_model_file = "{}_model.h5".format(FLAGS.dataset)
		model.save(full_path(saved_model_file))
		s3.upload(saved_model_file)

if __name__ == '__main__':
	tf.app.run()

