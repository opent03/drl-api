# Deep Reinforcement Learning API (drl_api)

drl_api is a (deep) RL repo with various implementations of DRL algorithms in **pytorch**. It is designed for quick prototyping and implementation of new algorithms, using its simple and intuitive API. 

It is loosely based on Nikolov's [rltf](https://github.com/nikonikolov/rltf). 

Maintained regularly, currently supports most Atari NoFrameskip environments and bsuite. 

## Train model
A highly intuitive example command:
```
python3 -m run.run_dqn_agent --env_id=BreakoutNoFrameskip-v4 --model=DDQN [--render] [--no-gpu]
```

## Load a saved model and evaluate
Models are saved automatically in drl_api/saves/. A highly intuitive example command:
```
python3 -m run.play --save_name=BreakoutNoFrameskip-v4-DDQN --rounds=50
```
