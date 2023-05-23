# Code for "Policy Regularization with Dataset Constraint for Offline Reinforcement Learning"

## Install dependency

```bash
pip install -r requirements.txt
```

Install the [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd d4rl
pip install -e .
```

## Run experiment

```bash
python main.py --env_id hopper-medium-v2 --seed 1024 --device cuda:0 --alpha 2.5 --beta 2.0 --k 1
```

## See result

```bash
tensorboard --logdir='./result'
```