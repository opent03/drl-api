import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from drl_api.models import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN_Model(Model):
    '''Model class, maintains an inner neural network, contains learn method'''

    def __init__(self, obs_shape, n_actions, eps, gamma, lr):

        assert len(obs_shape) == 3 or len(obs_shape) == 1

        super().__init__()

        self.gamma = gamma
        self.obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float32
        self.obs_shape = obs_shape
        self.act_dtype = torch.uint8
        self.act_shape = []
        self.n_actions = n_actions
        self.eps = eps
        self.lr = lr

        # Initialize networks
        self.Q_eval = DQN(in_dim=obs_shape, out_dim=n_actions, lr=lr, name='eval')
        self.Q_target = DQN(in_dim=obs_shape, out_dim=n_actions, lr=lr, name='target')
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def learn(self):
        pass

    def _conv_nn(self, x):
        pass

    def _dense_nn(self, x):
        pass

class DQN(nn.Module):
    ''' Basic Implementation of DQN '''
    def __init__(self, in_dim, out_dim, lr, name='eval', nntype='conv'):
        super(DQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc_in_dim = 5
        self.nntype = nntype

        # select architecture
        if self.nntype == 'dense':
            self.fc1 = nn.Linear(in_dim, 512)
            self.fc2 = nn.Linear(512,512)
            self.fc3 = nn.Linear(512, self.out_dim)
        elif self.nntype == 'conv':
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_dim, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=3),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1),
                nn.ReLU()
            )
            self.fc = nn.Sequential(
                nn.Linear(self.fc_in_dim, 512),
                nn.ReLU(),
                nn.Linear(512, self.out_dim)
            )
        else:
            print('Architecture invalid: {}'.format(self.nntype))
            exit(1)

        self.name = name
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(device)
        qvals = None
        if self.nntype == 'dense':
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            qvals = self.fc3(x)

        elif self.nntype == 'conv':
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            qvals = self.fc(x)

        else:
            print('Architecture invalid: {}'.format(self.nntype))
            exit(1)

        return qvals