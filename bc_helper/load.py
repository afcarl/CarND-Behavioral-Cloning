# /bin/python

import pandas as pd
import numpy as np
import os
from bc_helper.simulator_data import SimulatorData
from bc_helper.full_path import full_path

def load_simple_data():
	data_frame_folder = 'data_frames'
	data_frame_file = 'original'
	absolute_data_frame_folder = full_data_path(data_frame_folder)
	absolute_data_frame_file = full_data_path(data_frame_folder + "/" + data_frame_file)

	if os.path.isdir(absolute_data_frame_folder) == False:
		os.mkdir(absolute_data_frame_folder)

	if os.path.isfile(absolute_data_frame_file) == False:
		df = _all_original_data()
		df.to_csv(absolute_data_frame_file)

	return pd.read_csv(absolute_data_frame_file)

def load_starter_data():
	data_frame_folder = 'data_frames'
	data_frame_file = 'starter_data'

	absolute_data_frame_folder = full_data_path(data_frame_folder)
	absolute_data_frame_file = full_data_path(data_frame_folder + "/" + data_frame_file)

	if os.path.isdir(absolute_data_frame_folder) == False:
		os.mkdir(absolute_data_frame_folder)

	if os.path.isfile(absolute_data_frame_file) == False:
		df = _all_starter_data()
		df.to_csv(absolute_data_frame_file)

	return pd.read_csv(absolute_data_frame_file)

def load_smooth_data():
	data_frame_folder = 'data_frames'
	data_frame_file = 'smooth_steering'
	absolute_data_frame_folder = full_data_path(data_frame_folder)
	absolute_data_frame_file = full_data_path(data_frame_folder + "/" + data_frame_file)

	if os.path.isdir(absolute_data_frame_folder) == False:
		os.mkdir(absolute_data_frame_folder)

	if os.path.isfile(absolute_data_frame_file) == False:
		df = create_smooth_data_frame()
		df.to_csv(absolute_data_frame_file)

	return pd.read_csv(absolute_data_frame_file)

def _all_original_data():
	return _original_data_frame(['smooth', 'recovery'])

def _original_smooth_data():
	return _original_data_frame(['smooth'])

def _all_starter_data():
	return _original_data_frame(['starter_data'])

def _original_data_frame(folders):
	csv_name = 'driving_log.csv'
	columns = ['center','left','right','steering','throttle','brake','speed']
	frames = []
	for folder in folders:
		for subfolder in os.listdir(full_data_path(folder)):
			s_path = full_data_path("{}/{}".format(folder, subfolder))
			if os.path.isdir(s_path):
				frames.append(pd.read_csv(s_path + "/" + csv_name, names=columns)[['steering', 'center', 'left', 'right']])

	data_frame = pd.concat(frames, ignore_index=True)
	return data_frame.reindex()

def full_data_path(name):
	return full_path("data/{}".format(name))

def create_smooth_data_frame():
	df = _original_smooth_data()
	print('len(df):', len(df))
	new = []
	length = len(df)
	for index, row in df.iterrows():
		new.append(np.mean(df['steering'][max(index - 5, 0):min(index + 5, length)]))
	df['smooth_steering'] = new
	return df



