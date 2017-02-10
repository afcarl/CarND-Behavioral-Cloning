#/bin/python

import numpy as np
from sklearn.model_selection import ShuffleSplit
from keras.preprocessing.image import img_to_array, load_img

class SimulatorData(object):
	def __init__(self, data_frame, batch_size=32):
		self._df = data_frame
		left, right = ShuffleSplit(n_splits=2, test_size=.3, random_state=0).split(self._df)
		self.train_indices = left[0] # TODO: I don't understand why this works
		self.validation_indices = right[1]
		self.batch_size = batch_size

		self.max_label = self._df.max(numeric_only=True).values[0]
		self.min_label = self._df.min(numeric_only=True).values[0]
		self.num_train = len(self.train_indices)
		self.num_validation = len(self.validation_indices)
		self.feature_shape = self.img(0).shape

	def labels(self, indices):
		# label selection based on array of indices.
		return self._df['steering'].iloc[indices]

	def features(self, indices):
		# features selection based on array of indices.
		return [self.img(i) for i in indices]

	def img(self, index):
		return img_to_array(load_img(self._df['center'][index]))

	def train_labels(self):
		return self.labels(self.train_indices)

	def validation_labels(self):
		return self.labels(self.validation_indices)

	def train_generator(self):
		return BatchGenerator(self, is_train=True)

	def validation_generator(self):
		return BatchGenerator(self, is_train=False)

class BatchGenerator:
	def __init__(self, data, is_train=True):
		self.data = data
		self.batch_index = 0
		self.indices = data.train_indices if is_train else data.validation_indices

	def next(self):
		last_index = len(self.indices)
		start_index = self.batch_index
		end_index = min(self.batch_index + self.data.batch_size, last_index)
		self.batch_index = end_index
		return (np.array(self.data.features(self.indices[start_index:end_index])), np.array(self.data.labels(self.indices[start_index:end_index])))

	def __next__(self):
		return self.next()











