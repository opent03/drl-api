import os
import numpy as np
import torch

from abc import ABCMeta, abstractmethod

class Agent(metaclass=ABCMeta):
    '''Base abstract RL agent class'''

    def __init__(self,
                 model,
                 env_train,
                 env_eval,
                 env_name,
                 ):

        # Env data
        self.env_train = env_train
        self.env_eval = env_eval
        self.env_name = env_name

        # Model data
        self.model = model

        # Training data
        self.agent_step = 0

        # Run data
        self._terminate = False


    def eval_step(self):
        '''1-episode evaluation'''
        score = 0
        obs = self.env_eval.reset()
        done = False
        while not done:
            if self._terminate:
                break
            action = self.act_eval(obs)
            obs_, reward, done, info = self.env_eval.step(action)
            score += reward
            if done:
                obs_ = self.env_eval.reset()
            obs = obs_

        return score

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

    @abstractmethod
    def train(self, episodes):
        pass