from drl_api.agents import Agent

class dqn_agent(Agent):
    '''Base class for a Q-learning based agent. Uses target networks and experience replay.'''

    def __init__(self,
                 warm_up,
                 train_period,
                 target_update_period,
                 batch_size,
                 stop_step,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args,**kwargs)

        # Training data
        self.warm_up = warm_up  # Steps from which the training starts
        self.train_period = train_period
        self.stop_step = stop_step
        self.batch_size = batch_size

        self.target_update_period = target_update_period    # Period at which target network is updated
        self.replay_memory = None

