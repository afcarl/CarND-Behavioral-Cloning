#!/bin/python

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

from bc_helper.sync_driving_data import put_driving_data

put_driving_data()
