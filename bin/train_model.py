#!/bin/python 

import sys
import os

# add project root to path so bc_helper is accessable. 
project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

from bc_helper.load import load_simple_data
from bc_helper.load import load_starter_data
from bc_helper.load import load_augmented_starter_data
from bc_helper.load import load_final_data
from bc_helper.simulator_data import SimulatorData
from bc_helper.full_path import full_path
from bc_helper import s3

from keras.layers.normalization import BatchNormalization
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Train model from command line rather than a jupyter notebook.
flags.DEFINE_string('dataset', 'original', "Make bottleneck features this for dataset, one of 'original', 'starter_data', augmented_starter, or final")
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator.')
flags.DEFINE_integer('epochs', 30, 'The number of epochs.')
flags.DEFINE_boolean('save', False, 'Save the generated bottleneck model to S3.')
# load_model allows for iterative testing in the simulator. 
flags.DEFINE_boolean('load_model', False, 'Load previously trained model.')
flags.DEFINE_integer('initial_epoch', 0, 'The last trained epoch. Needed when using load_model param above.')

def main(_):
	# The following load_* functions returns a DataFrame. 
	# To allow for iterative testing, I never appened data to an existing dataframe. 
	# Instead, I create a new folder and added that data to the folder. The load_* 
	# functions concatentate certain sets of folders of csv's into one DataFrame
	# and return it. To see what folders are getting concatenated see bc_helper/load.py.

	# original: The first data I generated using the arrow keys. This data is not very good.
	if FLAGS.dataset == "original":
		data_frame = load_simple_data()
	# starter_data: The data provided by Udacity with the project.
	elif FLAGS.dataset == "starter_data":
		data_frame = load_starter_data()
	# augmented_starter: the starter data but augmented by flipping images and 
	# using the left and right canera images to "push" the car back towards the center of the lane.
	elif FLAGS.dataset == "augmented_starter":
		data_frame = load_augmented_starter_data()
	# final: The last dataset I created which happened to work. It includes all the same data as
	# augmented_starter plus data I added based on where the car crashed during testing. Namely,
	# the red caution areas, crossing the bridge, and data from the second track. 
	elif FLAGS.dataset == "final":
		data_frame = load_final_data()
	else:
		raise Exception("Unexpected dataset:", dataset)
		
	# SimulatorData is a small wrapper around the DataFrame and provides a generator interface
	# to use Keras as well as some helper methods for undertanding the contents of the DataFrame.
	data = SimulatorData(data_frame, FLAGS.batch_size)

	if FLAGS.load_model:
		model = load_model(args.model)
	else:
		# Model is based on NVIDIA driving paper:
		# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
		# - Crop
		# - Normalization
		# - CNN(s=2x2, k=5x5, d=24, activation='relu')
		# - CNN(s=2x2, k=5x5, d=36, activation='relu')
		# - CNN(s=2x2, k=5x5, d=48, activation='relu')
		# - CNN(s=1x1, k=3x3, d=64, activation='relu')
		# - CNN(s=1x1, k=3x3, d=64, activation='relu')
		# - FCC(d=100, activation='relu')
		# - Dropout(0.5)
		# - FCC(d=50, activation='relu')
		# - Dropout(0.5)
		# - FCC(d=10, activation='relu')
		# - Dropout(0.5)
		# - FCC(d=1)
		model = Sequential()
		model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=data.feature_shape))
		model.add(Lambda(lambda x: (x / 255.0) - 0.5))
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
		nb_val_samples=data.num_validation,
		initial_epoch=FLAGS.initial_epoch)

	print("Finished training model!")

	if FLAGS.save:
		saved_model_file = "{}_model.h5".format(FLAGS.dataset)
		model.save(full_path(saved_model_file))
		s3.upload(saved_model_file)

if __name__ == '__main__':
	tf.app.run()

