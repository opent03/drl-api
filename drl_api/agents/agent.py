import numpy as np
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


    def play(self, rounds, render=False):
        # a method for the agent to just play the game
        scores = []
        obs = self._format_img(self.env_eval.reset())
        for _ in range(rounds):
            score = 0
            done = False
            while not done:
                if render:
                    self.env_eval.render()
                action = self.act_eval(obs)
                # run action
                obs_, reward, done, info = self.env_eval.step(action)
                score += reward
                if done:
                    obs = self.env_eval.reset()
                    obs = self._format_img(obs)
                    break
                obs = self._format_img(obs_)
            scores.append(score)
        return sum(scores)/len(scores)  # average scores


    def _format_img(self, img):
        ''' changes the format into something pytorch will be happy about '''
        if len(img.shape) == 3:
        # current: W x H x C
        # change to: C x W x H
            return np.transpose(img, (2,0,1))
        else:
            # need to flatten
            return img.flatten()

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