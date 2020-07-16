import os
import numpy as np
import torch

from abc import ABCMeta, abstractmethod

class Agent(metaclass=ABCMeta):
    '''Base abstract RL agent class'''

    def __init__(self,
                 model,
                 env,
                 eval_period,
                 eval_len,
                 n_plays,

                 **model_kwargs
                 ):

        # Env data
        self.env = env

        # Model data
        self.model = model
        self.model_kwargs = model_kwargs

        # Training data
        self.agent_step = 0

        # Evaluation data
        self.eval_period = eval_period  # How often to take an evaluation run
        self.eval_len = eval_len        # How long does an evaluation run last
        self.eval_step = 0              # Current agent eval step

        # Run data
        self.n_plays = n_plays          # Evaluation related thingies
        self.eval_mode = n_plays > 0
        self._terminate = False


    def eval_step(self):
        if self.eval_len <= 0 or self.eval_period <= 0:
            return
        start_step = self.eval_step + 1
        stop_step = start_step + self.eval_len

        obs = self.env.reset()

        for _ in range(start_step, stop_step):
            if self._terminate:
                break
            action = self.act_eval(obs)
            obs_, reward, done, info = self.env.step(action)

            if done:
                obs_ = self.env.reset()
            obs = obs_

    @abstractmethod
    def train_step(self):
        '''Train method, to be implemented by agent'''
        pass


    @abstractmethod
    def act_train(self):
        '''Choose an action at training time'''
        pass

    @abstractmethod
    def act_eval(self):
        '''Choose an action at evaluation time'''
        pass