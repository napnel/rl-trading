import argparse
import json
import os
import pickle
from pathlib import Path

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.tune.logger import pretty_print

parser = argparse.ArgumentParser()
args = parser.parse_args()


def load_config():
    with open("config.json") as f:
        config = json.load(f)

    agent_config = config["agent_config"]
    model_config = config["model_config"]
    env_config = config["env_config"]
    return agent_config, model_config, env_config


def main():
    ray.init()

    agent_config = {}
    model_config = {"use_attention": True}
    env_config = {}

    config = {
        "env": "CartPole-v0",
        "env_config": env_config,
        "model": model_config,
    }
    config.update(agent_config)

    # analysis = tune.run()
    # Import the RL algorithm (Trainer) we would like to use.

    # Configure the algorithm.
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "CartPole-v0",
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": model_config,
        # Set up a separate evaluation worker set for the
        # `trainer.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
    }

    # Create our RLlib Trainer.
    agent = PPOTrainer(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(10):
        pretty_print(agent.train())

    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
    agent.evaluate()

    ray.shutdown()


if __name__ == "__main__":
    main()
