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
            if self.replay_memory.counter > self.batch_size:
                # feed in a random batch
                random_batch = self.replay_memory.sample(batch_size=self.batch_size)
                s, a, r, s_, t = self.feed_dict(random_batch)
                self.learn(s=s, a=a, r=r, s_=s_, t=t)
            obs = obs_

            # update agent step conuter
            self.agent_step += 1

            # update target network
            if self.agent_step % self.target_update == 0:
                self.model.replace_target_network()
                print('Step {}: Target Q-Net replaced!'.format(self.agent_step))


    def learn(self, *args, **kwargs):
        '''agent learns, determined by agent's model'''
        self.model.learn(*args, **kwargs)


    def act_train(self, state):
        '''Choose an action at training time'''
        if np.random.random() > self.eps:
            action = self.model.get_action(state)
        else:
            action = np.random.choice(self.model.n_actions)

        return action


    def act_eval(self, state):
        ''' Choose an action at evaluation time '''
        ''' We make it so it follows behavior policy '''
        return self.model.get_action(state)


    def get_env_specs(self, stack_frames):
        ''' gets env parameters '''
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