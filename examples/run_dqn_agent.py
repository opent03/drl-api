from drl_api.utils import args_parser

def make_agent():
    model_choices = ["DQN"]
    args_parser.parse_args(model_choices)

    # create environment


    # create model


    # create agent
    return

def main():
    dqn_agent = make_agent()

if __name__ == '__main__':
    main()