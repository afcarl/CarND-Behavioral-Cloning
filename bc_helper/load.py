# /bin/python

import pandas as pd
import numpy as np
import os
from bc_helper.simulator_data import SimulatorData
from bc_helper.full_path import full_path

# load_* functions return a DataFrame which ahs been cleaned for use with SimulatorData.

def _load_df(file, df_loader):
	data_frame_folder = 'data_frames'
	data_frame_file = file
	absolute_data_frame_folder = full_data_path(data_frame_folder)
	absolute_data_frame_file = full_data_path(data_frame_folder + "/" + data_frame_file)

	if os.path.isdir(absolute_data_frame_folder) == False:
		os.mkdir(absolute_data_frame_folder)

	if os.path.isfile(absolute_data_frame_file) == False:
		df = df_loader()
		df.to_csv(absolute_data_frame_file)

	return pd.read_csv(absolute_data_frame_file)

def load_simple_data():
	return _load_df('original', _all_original_data)

def load_smooth_data():
	return _load_df('smooth', create_smooth_data_frame)

def load_starter_data():
	return _load_df('starter', _all_starter_data)

def load_augmented_starter_data():
	return _load_df('augmented_starter', _create_augmented_starter_data)

def load_final_data():
	return _load_df('final', _create_final_data_set)

def _all_original_data():
	return _original_data_frame(['smooth', 'recovery'])

def _original_smooth_data():
	return _original_data_frame(['smooth'])

def _all_starter_data():
	return _original_data_frame(['starter_data'])

def _all_final_data():
	return _original_data_frame(['trouble_areas_mouse', 'track_2_mouse', 'starter_data'])

def _original_data_frame(folders):
	csv_name = 'driving_log.csv'
	columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
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

def _create_augmented_starter_data():
	steering_delta = 0.5
	df = _all_starter_data()
	left = []
	right = []
	center = []
	for index, row in df.iterrows():
		left.append(pd.Series([row['steering'] + steering_delta, row['left'], False]))
		left.append(pd.Series([row['steering'] + steering_delta, row['left'], True]))
		right.append(pd.Series([row['steering'] - steering_delta, row['right'], False]))
		right.append(pd.Series([row['steering'] - steering_delta, row['right'], True]))
		center.append(pd.Series([row['steering'], row['center'], False]))
		center.append(pd.Series([row['steering'], row['center'], True]))
	    
	left = pd.DataFrame(left)
	right = pd.DataFrame(right)
	center = pd.DataFrame(center)
	new = pd.concat([left, right, center])
	new.columns = ['steering', 'center', 'flip']
	return new

def _create_final_data_set():
	steering_delta = 0.4
	df = _all_final_data()
	left = []
	right = []
	center = []
	for index, row in df.iterrows():
		# left
		if row.isnull()[2] == False:
			left.append(pd.Series([row['steering'] + steering_delta, row['left'], False]))
			left.append(pd.Series([row['steering'] + steering_delta, row['left'], True]))

	    #right
		if row.isnull()[3] == False:
			right.append(pd.Series([row['steering'] - steering_delta, row['right'], False]))
			right.append(pd.Series([row['steering'] - steering_delta, row['right'], True]))

		#center
		center.append(pd.Series([row['steering'], row['center'], False]))
		center.append(pd.Series([row['steering'], row['center'], True]))
	    
	left = pd.DataFrame(left)
	right = pd.DataFrame(right)
	center = pd.DataFrame(center)
	new = pd.concat([left, right, center])
	new.columns = ['steering', 'center', 'flip']
	return new	




