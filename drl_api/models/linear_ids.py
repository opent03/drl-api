import numpy as np

from drl_api.models import Model
from drl_api import envs

class LinearIDS_Model(Model):
    ''' Implements the model class for LinearIDS Algorithm'''
    def __init__(self, env_specs, H, lmbda=1, feature_set=None, seed=1234):
        super().__init__()
        self.rng = np.random.RandomState(seed)

        self.n_states, self.n_actions = env_specs['n_states'], env_specs['n_actions']
        self.H = H

        self.lmbda = lmbda

        if feature_set is None:
            features = np.identity(self.n_states)
            self.feature_set = [features[i] for i in range(self.n_states)]  # highly questionable
        else:
            self.feature_set = feature_set

        # indexing over states
        self.state_index = {self.feature_set[i].tostring() : i for i in range(self.n_states)}
        






