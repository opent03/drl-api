'''
Basic module to load a bot and play a game
'''
from drl_api import envs, models, agents
import os
import drl_api
import argparse


def load_model(path, name, arch, stack):
    env_id = name
    print(env_id)
    env = envs.make_env(env_id, stack= stack)
    env_specs = envs.get_env_specs(env['env_train'], stack)
    model = drl_api.model_dict[arch](env_specs=env_specs,
                             eps=1,
                             gamma=0.99,
                             lr=2.5e-4,
                             gpu=True,
                             memory_size=2e4,
                             nntype='conv'
                             )


    agent = agents.DQN_agent(target_update=5e3,
                             batch_size=32,
                             learn_frequency=4,
                             env_train=env['env_train'],
                             env_eval=env['env_eval'],
                             env_name=env_id,
                             model=model
                             )

    agent.load_save(path)

    return agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', required=True, type=str)
    parser.add_argument('--rounds', default=50, type=int)
    parser.add_argument('--render', dest='render', action='store_true')
    return parser.parse_args()


def main():
    # do something here
    args = parse_args()
    saves_path = 'drl_api/saves'
    path = os.path.join(saves_path, args.save_name)
    tmp = args.save_name.split('-')
    name = '-'.join(tmp[:-1])
    arch = tmp[-1]
    agent = load_model(path, name, arch, stack=4)
    agent.play(rounds=args.rounds, render=args.render)

if __name__ == '__main__':
    main()