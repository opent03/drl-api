import torch
import numpy as np
from drl_api.models import DQN_Model

class DDQN_Model(DQN_Model):
    ''' Class for Double DQN '''
    def __init__(self, *args, **kwargs):
        super(DDQN_Model, self).__init__(*args, **kwargs)
        self.name = 'DDQN'

    def learn(self, batch):
        self.Q_eval.optimizer.zero_grad()
        batch_dict = self.process_batch(batch)
        for key in batch_dict:
            batch_dict[key] = torch.tensor(batch_dict[key]).to(self.Q_eval.device)
        batch_size = batch_dict['state'].shape[0]
        batch_index = np.arange(batch_size, dtype=np.int32)

        # dqn step
        q_eval = self.Q_eval.forward(batch_dict['state'])[batch_index, batch_dict['action']]  # q values only for the action taken
        q_next = self.Q_target.forward(batch_dict['next_state'])
        q_next_eval = self.Q_eval.forward(batch_dict['next_state'])
        q_next[batch_dict['terminal']] = 0.0
        q_next_eval[batch_dict['terminal']] = 0.0

        # q_next_eval = self.Q_eval.forward(batch_dict['next_state'])
        ddqn_idx = torch.argmax(q_next_eval, dim=1)
        q_target = batch_dict['reward'] + self.gamma*q_next[batch_index, ddqn_idx]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()