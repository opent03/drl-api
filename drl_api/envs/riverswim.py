'''
Courtesy of Andrei Lupu, andrei.lupu@mail.mcgill.ca
'''

import numpy as np
from drl_api.envs import LinearMDP

class RiverSwim(LinearMDP):
    """
    Implement the classic RiverSwim MDP, as described here in Sect. 6 of https://arxiv.org/pdf/1306.0940.pdf
    """

    def __init__(self, n=6, H=20, seed=1234):

        super().__init__(n_states=n, n_actions=2, H=H, stationary=True, start_states=[0], seed=seed)


    def seed(self, seed):
        """
        Reseed environment
        :param seed: new seed.
        """
        self.rng.seed(seed)


    def gen_feature_map(self):
        """Generate features for RiverSwim (tabular)"""
        self.map = np.identity(self.n_states)


    def gen_linear(self):
        """
        Generate transition matrices P(s'|s,a) and rewards r(s,a)
        """

        rewards = np.zeros(self.d)
        rewards[self.get_index(self.n_states-1, 0)] = 1     # Right end
        rewards[self.get_index(0, 1)] = 5/1000              # Left end

        matrix = self.gen_transition_matrix()

        self.P = [matrix]*self.H
        self.rewards = [rewards]*self.H


    def gen_transition_matrix(self):
        """Generate transition matrix as specified in paper"""

        matrix = np.zeros(shape=(self.d, self.n_states))

        # Set forward probs
        a = 0
        for s in range(self.n_states):
            row = np.zeros(self.n_states)

            if s == 0:
                # Left end
                row[0] = .4
                row[1] = .6
            elif s == self.n_states-1:
                # Right end
                row[-1] = .6
                row[-2] = .4
            else:
                # Middle states
                row[s] = .6
                row[s-1] = .05
                row[s+1] = .35

            matrix[self.get_index(s,a)] = row

        # Set backward probs
        a=1
        for s in range(self.n_states):
            row = np.zeros(self.n_states)
            if s == 0:
                # Left end
                row[0] = 1
            else:
                row[s-1] = 1

            matrix[self.get_index(s, a)] = row

        return matrix

    def step(self, a):
        """
        Take a step in the environment
        """
        done = False

        idx = self.get_index(self.state, a)

        prev_state = self.state

        # Sample next state
        probs = self.P[self.t][idx]
        self.state = self.rng.choice(np.arange(self.n_states), p=probs)

        # Get reward
        r = 0
        if prev_state == 0 and a == 1:
            # Left end, backwards
            r = 5/1000
        elif prev_state == self.n_states-1 and self.state == self.n_states-1:
            r = 1

        # Increment time
        self.t += 1
        if self.t == self.H:
            done = True
            self.reset()

        return self.get_features(self.state), r, done