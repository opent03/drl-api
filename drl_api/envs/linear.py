'''
Courtesy of Andrei Lupu, andrei.lupu@mail.mcgill.ca
'''

import numpy as np
from scipy import sparse

class LinearMDP:
    def __init__(self, n_states, n_actions, H, stationary=False, start_states=None, density=.25, seed=1234):
        self.rng = np.random.RandomState(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.d = n_states * n_actions # default dimensions, works with the theory
        self.H = H                    # horizon
        self.stationary = stationary
        self.start_states = start_states
        self.density = density

        self.generateMDP()


    def generateMDP(self):
        self.gen_linear()
        self.gen_feature_map()


    def gen_linear(self):
        if self.stationary:
            rewards = self.rng.uniform(-1,1, size=self.d)
            matrix = self.gen_transition_matrix()

            self.P = [matrix]*self.H            # stationary MDP
            self.rewards = [rewards]*self.H     # stationary MDP

        else:
            self.P = [self.gen_transition_matrix() for _ in range(self.H)]
            self.rewards = [self.rng.uniform(-1, 1, size=self.d) for _ in range(self.H)]


    def gen_transition_matrix(self):
        matrix = sparse.random(self.d, self.n_states, density=self.density, random_state=self.rng).toarray()
        # Eliminate rows summing to zero
        zero_rows = np.where(1 - matrix.any(axis=1))[0]
        for row in zero_rows:
            n_nonzero = self.rng.randint(2, 5)
            cols = self.rng.choice(np.arange(self.n_states), n_nonzero, replace=False)

            for col in cols:
                matrix[row, col] = self.rng.uniform(0, 1)

        # Eliminate columns summing to zero
        zero_cols = np.where(1 - matrix.any(axis=0))[0]
        for col in zero_cols:
            n_nonzero = self.rng.randint(2, 5)
            rows = self.rng.choice(np.arange(self.d), n_nonzero, replace=False)

            for row in rows:
                matrix[row, col] = self.rng.uniform(0, 1)

        # Normalize probabilities to sum to 1
        for i in range(self.d):
            matrix[i] = matrix[i] / np.sum(matrix[i])

        return matrix


    def get_feature_map(self):
        ''' Generate a linear feature map phi'''
        M = self.rng.uniform(0, 1, size=(self.n_states, self.n_states))
        for i in range(self.n_states):
            M[i] = M[i] / (np.linalg.norm(M[i]))

        self.map = M


    def get_features(self, s):
        return self.map[s]


    def get_index(self, s, a):
        return self.n_states*a + s


    def reset(self):
        """
        Reset episode.
        """

        self.t = 0
        if self.start_states is not None:
            self.state = self.rng.choice(self.start_states)
        else:
            self.state = self.rng.randint(self.n_states)

        return self.get_features(self.state)


    def step(self, a):
        """
        Take a step in the environment
        """
        done = False

        idx = self.get_index(self.state, a)

        # Sample next state
        probs = self.P[self.t][idx]
        self.state = self.rng.choice(np.arange(self.n_states), p=probs)

        # Get reward
        r = self.rewards[self.t][idx]

        # Increment time
        self.t += 1
        if self.t == self.H:
            done = True
            self.reset()

        return self.get_features(self.state), r, done




