from typing import Any, Dict

import ray
from ray.rllib.agents import Trainer

from rl_trading.backtest import backtest
from rl_trading.envs import create_env


def simulate(
    agent_class: Trainer,
    config: Dict[str, Any],
    checkpoint_path: str,
    mode: str = "eval",
    render: bool = True,
):
    print("===" * 15, "Simulate()", "===" * 15)

    if mode == "eval":
        config["env_config"] = config["evaluation_config"]["env_config"].copy()

    config.pop("evaluation_config")
    config["num_workers"] = 1
    config["logger_config"] = {"type": ray.tune.logger.NoopLogger}
    env = create_env(config["env_config"])
    agent: Trainer = agent_class(config=config)
    agent.restore(checkpoint_path)

    backtest(env, agent, debug=False)

    # episode_reward = 0
    # done = False
    # obs = env.reset()
    # while not done:
    #     action = agent.compute_single_action(obs, explore=False, clip_action=True)
    #     # action = agent.compute_action(obs)
    #     print(action)
    #     obs, reward, done, info = env.step(action)
    #     episode_reward += reward

    # if render:
    #     env.render()
