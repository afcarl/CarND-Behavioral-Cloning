#!/bin/python

import boto3
import os
import sys
import threading
from bc_helper.full_path import full_path

bucket_name = 'kd-carnd'
key_name = 'behavioral_cloning/'
region_name = 'us-east-2'

def upload(rel_path):
	bucket = boto3.resource('s3', region_name=region_name).Bucket(bucket_name)

	print("Uploading file", rel_path)
	bucket.upload_file(full_path(rel_path), key_name + rel_path, Callback=UploadProgressPercentage(rel_path))
	print("Finished uploading file", rel_path)

def download(rel_path):
	bucket = boto3.resource('s3', region_name=region_name).Bucket(bucket_name)

	print("Downloading file", rel_path)
	bucket.download_file(key_name + rel_path, full_path(rel_path), Callback=DownloadProgressPercentage(rel_path))
	print("Finished downloading file", rel_path)

class UploadProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

class DownloadProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._seen_so_far = 0
        self._lock = threading.Lock()
    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            sys.stdout.write(
                "\r%s --> %s bytes transferred" % (
                    self._filename, self._seen_so_far))
            sys.stdout.flush()

