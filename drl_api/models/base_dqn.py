import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from drl_api.models import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN_Model(Model):
    '''Model class, maintains an inner neural network, contains learn method'''

    def __init__(self, obs_shape, n_actions, eps, gamma):

        assert len(obs_shape) == 3 or len(obs_shape) == 1

        super().__init__()

        self.gamma = gamma
        self.obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float32
        self.obs_shape = obs_shape
        self.act_dtype = torch.uint8
        self.act_shape = []
        self.n_actions = n_actions
        self.eps = eps


    def learn(self):
        pass

    def _conv_nn(self, x):
        pass

    def _dense_nn(self, x):
        pass

class DQN(nn.Module):
    ''' Basic Implementation of DQN '''
    def __init__(self, in_dim, out_dim, lr, name='eval'):
        super(DQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50, 2)
        self.name = name
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(device)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        qvals = self.fc3(x)
        return qvals

