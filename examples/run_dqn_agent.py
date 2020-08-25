from drl_api.utils import args_parser
from drl_api import envs, agents, models
import drl_api
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

    args = args_parser.parse_args(drl_api.model_choices)

    # create environments
    env = envs.make_env(args.env_id, stack=stack)
    env_specs = envs.get_env_specs(env['env_train'], stack)
    nntype = 'conv' if 'NoFrameskip' in args.env_id else 'dense'
    env_name = args.env_id.replace('/','')
    # create model
    model = drl_api.model_dict[args.model](
                             env_specs=env_specs,
                             eps=args.eps,
                             gamma=args.gamma,
                             lr=args.lr,
                             memory_size=args.mem_size,
                             gpu=args.gpu,
                             nntype=nntype,
                             eps_decay=args.eps_decay
                             )

    # create agent
    agent = agents.DQN_agent(target_update=5e3,
                             batch_size=args.batch_size,
                             learn_frequency=4,
                             env_train=env['env_train'],
                             env_eval=env['env_eval'],
                             env_name = env_name,
                             model=model
                             )
    return agent, args

def main():
    agent, args = make_agent(stack=4)
    # train loop
    print('Training ' + agent.model.name)
    agent.train(episodes=args.episodes, render=args.render)


if __name__ == '__main__':
    main()