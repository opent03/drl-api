from drl_api.envs.atari import wrap_deepmind_atari
import numpy as np

def get_env_specs(env, stack_frames):
    ''' gets env parameters '''
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    obs_shape = list(obs_shape)

    if len(obs_shape) == 3:
        assert stack_frames > 1
        obs_dtype = np.uint8
        obs_len = stack_frames
    else:
        obs_dtype = np.float32
        obs_len = 1

    return obs_shape, obs_dtype, obs_len, n_actions