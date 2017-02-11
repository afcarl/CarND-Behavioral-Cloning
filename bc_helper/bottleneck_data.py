#!/bin/python

from keras.applications.vgg16 import VGG16
from keras.layers import Input
from bc_helper.load import load_simple_data, load_smooth_data
from bc_helper.simulator_data import SimulatorData
import pickle
import os
from bc_helper.full_path import full_path
from bc_helper import s3

def files(dataset, batch_size):
	train_output_file = "{}_{}_{}.p".format(dataset, batch_size, 'bottleneck_features_train')
	validation_output_file = "{}_{}_{}.p".format(dataset, batch_size, 'bottleneck_features_validation')
	return (train_output_file, validation_output_file)

def load_bottleneck_model(dataset, batch_size=32):
	train_output_file, validation_output = files(dataset, batch_size)

	train_output_file_full = full_path(train_output_file)
	validation_output_file_full = full_path(validation_output_file)

	if os.path.isfile(train_output_file_full) == False or os.path.isfile(validation_output_file_full) == False:
		download_bottleneck_model(dataset, batch_size)

	f = open(train_output_file_full, mode='rb')
	train = pickle.load(f)
	f = open(validation_output_file_full, mode='rb')
	validation = pickle.load(f)
	return (train, validation)

def download_bottleneck_model(dataset, batch_size):
	train_output_file, validation_output = files(dataset, batch_size)
	s3.download(train_output_file)
	s3.download(validation_output_file)

def create_bottleneck_model(dataset, batch_size):
	if dataset == 'original':
		data = SimulatorData(load_simple_data(), batch_size)
	elif dataset == 'smooth':
		data = SimulatorData(load_smooth_data(), batch_size)
	else:
		raise Exception("Unexpected dataset:", dataset)

	train_output_file, validation_output_file = files(dataset, batch_size)

	print("Saving to ...")
	print(train_output_file)
	print(validation_output_file)

	model = VGG16(input_tensor=Input(shape=data.feature_shape), include_top=False)

	print('Bottleneck training')
	bottleneck_features_train = model.predict_generator(data.train_generator(), data.num_train)
	pickle_data = { 'features': bottleneck_features_train, 'labels': data.train_labels() }
	pickle.dump(pickle_data, open(train_output_file, 'wb'))

	print('Bottleneck validation')
	bottleneck_features_validation = model.predict_generator(data.validation_generator(), data.num_validation)
	pickle_data = { 'features': bottleneck_features_validation, 'labels': data.validation_labels() }
	pickle.dump(pickle_data, open(validation_output_file, 'wb'))

def save_bottleneck_model(dataset, batch_size):
	train_output_file, validation_output = files(dataset, batch_size)
	s3.upload(train_output_file)
	s3.upload(validation_output_file)
