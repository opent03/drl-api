'''
Basic module to load a bot and play a game
'''
from drl_api import envs, models, agents
def load_model(path, name,  stack):
    env_id = name
    print(env_id)
    env = envs.make_env(env_id, stack= stack)
    env_specs = envs.get_env_specs(env['env_train'], stack)
    model = models.DQN_Model(env_specs=env_specs,
                             eps=1,
                             gamma=0.99,
                             lr=2.5e-4,
                             gpu=True,
                             nntype='conv'
                             )

    agent = agents.DQN_agent(target_update=5e3,
                             batch_size=32,
                             memory_size=2e4,
                             learn_frequency=4,
                             env_train=env['env_train'],
                             env_eval=env['env_eval'],
                             env_name=env_id,
                             model=model
                             )

    agent.load_save(path)

    return agent


def main():
    # do something here
    path = 'drl_api/saves/BreakoutNoFrameskip-v4-_DQN'
    name = 'BreakoutNoFrameskip-v4'         # i don't care
    agent = load_model(path, name, stack=4)
    agent.play(rounds=50)

if __name__ == '__main__':
    main()