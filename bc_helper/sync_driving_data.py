# /bin/python

import os
import zipfile
from bc_helper import s3
from bc_helper.full_path import full_path

# Used to push and pull driving data to S3. Useful when working between AWS and local machine. 
# Used in ./bin/get_data.py and ./bin/put_data.py.

def zipdir(path, ziph):
	# ziph is zipfile handle
	for root, dirs, files in os.walk(path):
		for file in files:
			ziph.write(os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

zipfile_name = 'driving_data.zip'
zipfile_path = full_path(zipfile_name)

def put_driving_data():		
	data_folder = full_path('data')

	print("Zipping folder", data_folder)
	zf = zipfile.ZipFile(zipfile_path, "w")
	zipdir(data_folder, zf)
	zf.close()
	print("Finished zipping folder", data_folder)

	s3.upload(zipfile_name)

def get_driving_data():
	s3.download(zipfile_name)

	print("Unzipping file", zipfile_name)
	zip_ref = zipfile.ZipFile(zipfile_path, 'r')
	zip_ref.extractall(full_path(""))
	zip_ref.close()
	print("Finished unzipping file", zipfile_name)


