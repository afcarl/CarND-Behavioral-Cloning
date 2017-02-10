#!/bin/python

def full_path(name):
	base_dir_name = "CarND-Behavioral-Cloning"
	base_dir_list = os.getcwd().split("/")
	i = base_dir_list.index(base_dir_name)
	return "/".join(base_dir_list[0:i+1]) + "/" + name
