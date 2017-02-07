class Batch:
    def __init__(self, batch_size, features, labels):
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
    
    def iterator(self):
        return BatchIterator(self)
    
    def next(self, start):
        if self.batch_size == 0:
            raise StopIteration
        if start >= len(self.features):
            raise StopIteration 
        else:
            end = min(start + self.batch_size, len(self.features))
            return self.features[start:end], self.labels[start:end]

class BatchIterator:
    def __init__(self, batch):
        self.batch_index = 0
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.batch.next(self.batch_index)
        except StopIteration:
            raise StopIteration
        self.batch_index = self.batch_index + self.batch.batch_size
        return result
