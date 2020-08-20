import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from drl_api.models import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN_Model(Model):
    '''Model class, maintains an inner neural network, contains learn method'''

    def __init__(self,
                 env_specs,
                 eps,
                 gamma,
                 lr,
                 min_eps=0.1,
                 eps_decay=1e-5,
                 gpu=True,
                 nntype='conv'):


        super().__init__()

        self.gamma = gamma
        self.env_specs = env_specs
        self.obs_dtype = torch.uint8 if len(env_specs['obs_shape']) == 3 else torch.float32
        self.obs_shape = env_specs['obs_shape']
        self.act_dtype = torch.uint8
        self.n_actions = env_specs['n_actions']
        self.in_dims = env_specs['in_dims']
        print(eps)
        self.eps = _eps(eps, min_eps, eps_decay)
        self.lr = lr
        self.gpu = gpu
        self.nntype = nntype
        # Initialize networks
        self.Q_eval = _DQN(in_dim=self.in_dims, out_dim=self.n_actions, lr=lr, name='eval',gpu=self.gpu, nntype=self.nntype)          # get channel component
        self.Q_target = _DQN(in_dim=self.in_dims, out_dim=self.n_actions, lr=lr, name='target', gpu=self.gpu, nntype=self.nntype)      # get channel component
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_eval.to(self.Q_eval.device)
        self.Q_target.to(self.Q_target.device)


    def learn(self, **kwargs):
        ''' basic DQN learn '''
        self.Q_eval.optimizer.zero_grad()

        # convert to torch tensor types, variables are local
        batch_dict = dict((k,torch.tensor(v).to(self.Q_eval.device)) for k, v in kwargs.items())
        batch_size = batch_dict['s'].shape[0]
        batch_index = np.arange(batch_size, dtype=np.int32)

        # dqn step
        q_eval = self.Q_eval.forward(batch_dict['s'])[batch_index, batch_dict['a']]
        q_next = self.Q_target.forward(batch_dict['s_'])
        q_next[batch_dict['t']] = 0.0
        q_target = batch_dict['r'] + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

    def get_action(self, state):
        ''' passes action through net and do some argmax thingies '''
        state = torch.tensor(state, dtype=torch.float32).to(self.Q_eval.device)
        q_vals = self.Q_eval(state)
        return torch.argmax(q_vals).item()

    def replace_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def load_save(self, path):
        self.Q_eval.load_state_dict(torch.load(path))
        self.Q_target.load_state_dict(torch.load(path))



class _DQN(nn.Module):
    ''' Basic Implementation of common DQN nets '''
    def __init__(self, in_dim, out_dim, lr, name='eval', nntype='conv', gpu=True):
        super(_DQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc_in_dim = 2048       # to be used with CNN
        self.nntype = nntype

        # select architecture
        if self.nntype == 'dense':
            self.fc = nn.Sequential(
                nn.Linear(self.in_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.out_dim)
            )
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
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        qvals = None
        if self.nntype == 'dense':
            qvals = self.fc(x)

        elif self.nntype == 'conv':
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            qvals = self.fc(x)

        else:
            print('Architecture invalid: {}'.format(self.nntype))
            exit(1)

        return qvals

class _eps:
    ''' linear epsilon scheduler '''
    def __init__(self, eps, min_eps, decay):
        self.eps = eps
        self.min_eps = min_eps
        self.decay = decay

    def get_eps(self):
        ''' get epsilon + decay '''
        self.eps = max(self.min_eps, self.eps - self.decay)
        return self.eps

    def get_eps_no_decay(self):
        return self.eps