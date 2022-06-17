import pathlib
from typing import Any, Dict

from ray.rllib.agents import a3c, ddpg, dqn, impala, ppo, sac
from ray.rllib.agents.a3c import a2c
from ray.rllib.agents.ppo import appo


def get_agent_class(algo: str, _config: Dict[str, Any] = None):

    if algo == "DQN":
        agent = dqn.DQNTrainer
        config = dqn.DEFAULT_CONFIG.copy()

    elif algo == "A2C":
        agent = a2c.A2CTrainer
        config = a2c.A2C_DEFAULT_CONFIG.copy()

    elif algo == "A3C":
        agent = a3c.A3CTrainer
        config = a3c.DEFAULT_CONFIG.copy()

    elif algo == "PPO":
        agent = ppo.PPOTrainer
        config = ppo.DEFAULT_CONFIG.copy()

    elif algo == "SAC":
        agent = sac.SACTrainer
        config = sac.DEFAULT_CONFIG.copy()
        config["optimization"]["actor_learning_rate"] = _config["lr"]
        config["optimization"]["critic_learning_rate"] = _config["lr"]
        config["optimization"]["entropy_learning_rate"] = _config["lr"]
        config["rollout_fragment_length"] = 4
        config["target_network_update_freq"] = 256
        config["tau"] = 1e-3
        config["Q_model"] = {
            "custom_model": _config["model"]["custom_model"],
            "fcnet_hiddens": _config["model"]["fcnet_hiddens"],
        }
        config["policy_model"] = {
            "custom_model": _config["model"]["custom_model"],
            "fcnet_hiddens": _config["model"]["fcnet_hiddens"],
        }
        _config.pop("model")

    elif algo == "DDPG":
        agent = ddpg.DDPGTrainer
        config = ddpg.DEFAULT_CONFIG.copy()

    elif algo == "APPO":
        agent = ppo.APPOTrainer
        config = appo.DEFAULT_CONFIG.copy()

    elif algo == "IMPARA":
        agent = impala.ImpalaTrainer
        config = impala.DEFAULT_CONFIG.copy()

    else:
        raise ValueError
    return agent, config
