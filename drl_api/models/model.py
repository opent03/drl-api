import torch

from abc import ABCMeta


class Model(metaclass=ABCMeta):
    '''Abstract model class, inherited by all models'''

    def __init__(self):

        # Important properties
        self.obs_dtype = None
        self.obs_shape = None
        self.act_dtype = None
        self.act_shape = None
