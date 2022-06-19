from pathlib import Path
from pprint import pprint

import gym
from ray import tune
from ray.rllib.agents import Trainer, cql, dqn, pg, ppo, sac

DATA_PATH = Path("./data/BTCUSDT/").resolve()
TMP_PATH = Path("./tmp/").resolve()
EXPERIENCE_PATH = Path("./experience/").resolve()
DATA_PATH.mkdir(exist_ok=True)
TMP_PATH.mkdir(exist_ok=True)
EXPERIENCE_PATH.mkdir(exist_ok=True)

CONFIG = {
    "env": "MountainCar-v0",
    "framework": "torch",
}


def online_learning(config: dict, output: str = str(EXPERIENCE_PATH)):
    config["output"] = output
    agent = ppo.APPOTrainer(config=config)
    num_iterations = 300
    for i in range(num_iterations):
        results = agent.train()
        episode = results.get("episodes_total")
        step = results.get("timesteps_total")
        reward = results.get("episode_reward_mean")
        print(f"Iter {i} | Episode {episode} | Step {step} | Reward {reward}")
        if step >= 100000:
            break


def offline_learning(config: dict, input: str):
    agent = cql.CQLTrainer(config=config)

    env = gym.make(env)
    print(env.action_space, env.observation_space)
    obs = env.reset()
    done = False
    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()

def evaluation():
    pass


def main():
    online_learning(CONFIG)
    offline_learning(CONFIG)


if __name__ == "__main__":
    main()

