import torch
from drl_api.models import DQN_Model

class BaseBootstrapDQN(DQN_Model):
    def __init__(self, huber_loss, n_heads, **kwargs):
        super().__init__(**kwargs)

        self.huber_loss = huber_loss
        self.n_heads = n_heads

