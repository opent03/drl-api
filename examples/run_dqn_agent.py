from drl_api.utils import args_parser
from drl_api import envs, agents, memory, models, utils
import gym
''' 
    parser has arguments:
    env_id
    model
    batch_size
    eps
    gamma
    lr
'''


def make_agent(stack=4):

    model_choices = ["DQN"]
    args = args_parser.parse_args(model_choices)

    # create environments
    env = envs.wrap_deepmind_atari(args.env_id, stack=stack)
    obs_shape, obs_dtype, obs_len, n_actions = envs.get_env_specs(env['env_train'], stack)

    # create model
    model = models.DQN_Model(obs_shape=obs_shape,
                             n_actions=n_actions,
                             eps=args.eps,
                             gamma=args.gamma,
                             lr=args.lr,
                             gpu=args.gpu
                             )

    # create agent
    agent = agents.DQN_agent(target_update=5e3,
                             batch_size=args.batch_size,
                             memory_size=2e4,
                             learn_frequency=4,
                             env_train=env['env_train'],
                             env_eval=env['env_eval'],
                             env_name = args.env_id,
                             model=model
                             )
    return agent, args

def main():
    agent, args = make_agent(stack=4)
    # train loop
    agent.train(episodes=args.episodes, render=args.render)


if __name__ == '__main__':
    main()