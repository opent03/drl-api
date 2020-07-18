import numpy as np

from drl_api.agents import Agent
from drl_api.memory import ReplayMemory


class DQN_agent(Agent):
    '''Base class for a DQN agent. Uses target networks and experience replay.'''

    def __init__(self,
                 target_update,
                 batch_size,
                 eps,
                 stop_step,
                 memory_size=1e6,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args,**kwargs)

        # Training data
        self.batch_size = batch_size
        self.eps = eps
        self.stop_step = stop_step  # Step at which training stops

        self.target_update = target_update    # Period at which target network is updated
        self.replay_memory = ReplayMemory(memory_size)


    def train_step(self):
        '''Sample a 1-episode trajectory and learn simultaneously'''

        obs = self.env.reset()
        done = False
        while not done:
            if self._terminate:
                break

            # get action
            action = self.act_train(obs)

            # run action
            obs_, reward, done, info = self.env.step(action)

            # store transition in replay buffer
            self.replay_memory.push()

            # learn something
            self.learn()
            obs = obs_


    def learn(self):
        '''agent learns, determined by agent's model'''
        self.model.learn()

    def act_train(self, state):
        '''Choose an action at training time'''
        pass


    def act_eval(self, state):
        '''Choose an action at evaluation time'''
        pass

    def get_env_specs(self, stack_frames):
        n_actions = self.env.action_space.n
        obs_shape = self.env.observation_space.shape
        obs_shape = list(obs_shape)

        if len(obs_shape) == 3:
            assert stack_frames > 1
            obs_dtype = np.uint8
            obs_len = stack_frames
        else:
            obs_dtype = np.float32
            obs_len = 1

        return obs_shape, obs_dtype, obs_len, n_actions