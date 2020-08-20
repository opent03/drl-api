import torch
import torch.nn as nn
import numpy as np
from drl_api.models import DQN_Model
from drl_api.models import _DQN

class BootstrapDQN(DQN_Model):
    ''' Base Bootstrap DQN Model '''
    def __init__(self, huber_loss, n_heads, **kwargs):
        super().__init__(**kwargs)

        self.huber_loss = huber_loss
        self.n_heads = n_heads

        self.Q_eval = _BootDQN(n_heads=self.n_heads,
                               n_dim=self.obs_shape[2],
                               out_dim=self.n_actions,
                               lr=self.lr,
                               name='eval',
                               gpu=self.gpu)

        self.Q_target = _BootDQN(n_heads=self.n_heads,
                               n_dim=self.obs_shape[2],
                               out_dim=self.n_actions,
                               lr=self.lr,
                               name='eval',
                               gpu=self.gpu)


class _BootDQN(_DQN):
    ''' Bootstrapped Deeeeep Q Network '''
    def __init__(self, n_heads, *args, **kwargs):
        self.n_heads = n_heads
        super(_BootDQN, self).__init__(*args, **kwargs)

        # Make bootstrapped heads
        self.heads = [self.make_head(self.in_dim, self.out_dim) for _ in range(self.n_heads)]
        self.cur_head = None


    def make_head(self, in_dim, out_dim):
        ''' Creates a bootstrap head '''
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )


    def forward(self, x):
        pass


    def set_head(self, n):
        self.cur_head = n