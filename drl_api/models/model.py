import torch

from abc import ABCMeta, abstractmethod



class Model(metaclass=ABCMeta):
    '''Abstract model class, inherited by all models'''

    def __init__(self):

        # Important properties
        self.obs_dtype = None
        self.obs_shape = None
        self.act_dtype = None
        self.act_shape = None

    def process_batch(self, batch):
        ''' convert a list of named tuples to batch dictionary '''
        batch_dict = {}
        fields = batch[0]._fields
        for field in fields:
            batch_dict[field] = []
            for i in range(len(batch)):
                batch_dict[field].append(getattr(batch[i], field)) # ah yes minuscule brain approach here

        return batch_dict

    @abstractmethod
    def learn(self, batch):
        pass