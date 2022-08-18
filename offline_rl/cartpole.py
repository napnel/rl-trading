import argparse
from pathlib import Path
from pprint import pprint

import gym
from ray.rllib.agents import a3c, cql, dqn, pg

CONFIG = {
    "env": "MountainCar-v0",
    "framework": "torch",
}

EXPERIENCE_PATH = Path.home() / "experience" / CONFIG["env"]
EXPERIENCE_PATH.mkdir(exist_ok=True, parents=True)


def online_learning(config: dict, output_path: str = str(EXPERIENCE_PATH)):
    config["output"] = output_path
    # agent = ppo.APPOTrainer(config=config)
    # agent = pg.PGTrainer(config=config)
    agent = dqn.DQNTrainer(config=config)
    num_iterations = 300
    for i in range(num_iterations):
        results = agent.train()
        episode = results.get("episodes_total")
        step = results.get("timesteps_total")
        reward = results.get("episode_reward_mean")
        print(f"Iter {i} | Episode {episode} | Step {step} | Reward {reward}")
        if step >= 100000:
            break


def offline_learning(config: dict, input_path: Path = EXPERIENCE_PATH):
    data_files = [str(path) for path in input_path.glob("*.json")]
    print(data_files)
    config["input"] = data_files
    config["num_workers"] = 0  # Run locally.
    config["horizon"] = 200
    config["soft_horizon"] = True
    config["no_done_at_end"] = True
    config["n_step"] = 3
    config["bc_iters"] = 0
    # Set up evaluation.
    config["evaluation_num_workers"] = 1
    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 3
    # Evaluate on actual environment.
    config["evaluation_config"] = {"input": "sampler"}
    agent = cql.CQLTrainer(config=config)

    env = gym.make(config["env"])
    print(env.action_space, env.observation_space)
    obs = env.reset()
    done = False
    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()


def evaluation():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--offline", action="store_true")

    if parser.parse_args().offline:
        offline_learning(CONFIG)

    else:
        online_learning(CONFIG)
