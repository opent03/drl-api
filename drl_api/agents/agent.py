import os
import numpy as np
import torch

from abc import ABCMeta, abstractmethod

class Agent(metaclass=ABCMeta):
    '''Base abstract RL agent class'''

    def __init__(self,
                 model,
                 env,
                 ):

        # Env data
        self.env = env

        # Model data
        self.model = model

        # Training data
        self.agent_step = 0

        # Run data
        self._terminate = False


    def eval_step(self):
        '''1-episode evaluation'''

        obs = self.env.reset()
        done = False
        while not done:
            if self._terminate:
                break
            action = self.act_eval(obs)
            obs_, reward, done, info = self.env.step(action)

            if done:
                obs_ = self.env.reset()
            obs = obs_


    def feed_dict(self, batch):
        state_batch, action_batch, reward_batch, new_state_batch, terminal_batch = [], [], [], [], []
        for i in range(len(batch)):
            # assert batch[i].state.shape == (100,), batch[i].next_state.shape == (100,)
            state_batch.append(batch[i].state)
            action_batch.append(batch[i].action)
            reward_batch.append(batch[i].reward)
            new_state_batch.append(batch[i].next_state)
            terminal_batch.append(batch[i].terminal)
        return state_batch, action_batch, reward_batch, new_state_batch, terminal_batch


    @abstractmethod
    def train_step(self):
        '''One episode train step, samples trajectory and simultaneously learn'''
        pass

    @abstractmethod
    def learn(self):
        '''Agent learns something from model'''
        pass

    @abstractmethod
    def act_train(self, state):
        '''Choose an action at training time'''
        pass


    @abstractmethod
    def act_eval(self, state):
        '''Choose an action at evaluation time'''
        pass
