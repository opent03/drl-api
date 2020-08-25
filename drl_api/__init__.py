from drl_api import agents
from drl_api import models
from drl_api import memory
from drl_api import utils

model_choices = ['DQN', 'DDQN', 'BootstrapDQN']
model_dict = {'DQN': models.DQN_Model, 'DDQN': models.DDQN_Model, 'BootstrapDQN': models.BootstrapDQN_Model}

