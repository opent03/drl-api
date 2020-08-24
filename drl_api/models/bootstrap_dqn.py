import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from drl_api.models import DQN_Model
from drl_api.models import _DQN
from drl_api.memory import ReplayMemory

class BootstrapDQN(DQN_Model):
    ''' Base Bootstrap DQN Model '''
    def __init__(self, huber_loss, n_heads, p=0.5, **kwargs):
        super().__init__(**kwargs)

        self.huber_loss = huber_loss
        self.n_heads = n_heads
        self.p = p
        self.replay_memory = ReplayMemory(self.memory_size,
                                          namedtuple('Transition',
                                                     ('state',
                                                      'action',
                                                      'reward',
                                                      'next_state',
                                                      'terminal',
                                                      'mask'))
                                          )




    def init_networks(self):
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
                                 name='target',
                                 gpu=self.gpu)

        # weights initialization
        self.Q_eval.apply(self.Q_eval.init_weights)

        # eval-target setup
        self.replace_target_network()
        self.Q_eval.to(self.Q_eval.device)
        self.Q_target.to(self.Q_target.device)


    def learn(self, batch):
        self.Q_eval.optimizer.zero_grad()
        batch_dict = self.process_batch(batch)
        for key in batch_dict:
            batch_dict[key] = torch.tensor(batch_dict[key]).to(self.Q_eval.device)

        batch_size = batch_dict['state'].shape[0]
        batch_index = np.arange(batch_size, dtype=np.int32)

        # ddqn step
        q_eval = self.Q_eval.forward(batch_dict['state'])[batch_index, batch_dict['action']]
        q_next = self.Q_target.forward(batch_dict['state'])
        q_next[batch_dict['terminal']] = 0.0
        q_next_eval = self.Q_eval.forward(batch_dict['next_state'])
        #acts = torch.argmax(self.)
        #q_target = batch_dict['reward'] + self.gamma * torch.max(2)



    def store_transition(self, *args):
        ''' Implements Bernoulli mask '''
        args = args + (np.random.binomial(1, self.p, self.n_heads),)
        self.replay_memory.push(args)


class _BootDQN(_DQN):
    ''' Bootstrapped Deeeeep Q Network '''
    def __init__(self, n_heads, huber=False, *args, **kwargs):
        self.n_heads = n_heads
        super(_BootDQN, self).__init__(*args, **kwargs)

        # Make bootstrapped heads
        self.heads = [self.make_head(self.in_dim, self.out_dim) for _ in range(self.n_heads)]
        self.cur_head = None    # index of current head, type: int
        self.huber_loss = huber
        # redefine loss function
        self.loss = nn.SmoothL1Loss() if self.huber_loss else nn.MSELoss()


    def make_head(self, in_dim, out_dim):
        ''' Creates a bootstrap head '''
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )


    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        q_vals = self.heads[self.cur_head](x)

        return q_vals


    def set_head(self, n):
        self.cur_head = n