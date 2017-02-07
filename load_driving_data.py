# /bin/python

import os
import boto3

def full_path(name):
	base_dir_name = "CarND-Behavioral-Cloning"
	base_dir_list = os.getcwd().split("/")
	i = base_dir_list.index(base_dir_name)
	return "/".join(base_dir_list[0:i+1]) + "/" + name

if __name__ == '__main__':
	data_folder = full_path('data')

	if os.path.isdir(data_folder) == False:
		os.mkdir(data_folder)

	smooth_folder = full_path('data/smooth')

	if os.path.isdir(smooth_folder) == False:
		os.mkdir(smooth_folder)
		#get data from s3

	recovery_folder = full_path('recovery')
	if os.path.isdir(recovery_folder) == False:
		os.mkdir(recovery_folder)
		# get s3 data
