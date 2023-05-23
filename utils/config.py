import torch
import numpy as np
import argparse
import gym
import d4rl


def get_config(algorithm="PRDC"):
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default=algorithm)  # Policy name
    parser.add_argument(
        "--env_id", default="walker2d-random-v2"
    )  # OpenAI gym environment name
    parser.add_argument(
        "--seed", default=1024, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--eval_freq", default=1000, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--save_model_freq", default=500000, type=int
    )  # How often (time steps) we save model
    parser.add_argument(
        "--max_timesteps", default=1e6, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--save_model", default=False, action="store_false"
    )  # Save model and optimizer parameters
    parser.add_argument(
        "--load_model", default="default"
    )  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--device", default="cuda", type=str)  # Use gpu or cpu
    parser.add_argument("--info", default="default")  # Additional information
    # TD3
    parser.add_argument("--actor_lr", default=3e-4, type=float)  # Actor learning rate
    parser.add_argument("--critic_lr", default=3e-4, type=float)  # Critic learning rate
    parser.add_argument(
        "--expl_noise", default=0.1
    )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--noise_clip", default=0.5
    )  # Range to clip target policy noise
    parser.add_argument(
        "--policy_freq", default=2, type=int
    )  # Frequency of delayed policy updates
    parser.add_argument(
        "--normalize", default=True, action="store_false"
    )  # Whether to normalize the states
    # KD_TREE
    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--k", default=1, type=int)
    args = parser.parse_args()

    env = gym.make(args.env_id)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "device": args.device,
        # TD3
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # KD_TREE
        "alpha": args.alpha,
        "beta": args.beta,
        "k": args.k,
    }

    return args, env, kwargs


def save_config(args, arg_path):
    argsDict = args.__dict__
    with open(arg_path, "w") as f:
        for key, value in argsDict.items():
            f.write(key + " : " + str(value) + "\n")
