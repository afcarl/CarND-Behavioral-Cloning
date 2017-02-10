# /bin/python

import os
import zipfile
from bc_helper import s3

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
        	ziph.write(os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

zipfile_name = 'driving_data.zip'
zipfile_path = full_path(zipfile_name)
data_folder = full_path('data')

def put_driving_data():		
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
	zip_ref.extractall(data_folder)
	zip_ref.close()
	print("Finished unzipping file", zipfile_name)


