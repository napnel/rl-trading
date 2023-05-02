import pathlib
from typing import Any, Dict

from ray.rllib.algorithms import a2c, a3c, appo, ddpg, dqn, impala, ppo, sac


def get_agent_class(algo: str, _config: Dict[str, Any] = None):

    if algo == "DQN":
        agent = dqn.DQN
        config = dqn.DEFAULT_CONFIG.copy()

    elif algo == "A2C":
        agent = a2c.A2C
        config = a2c.A2C_DEFAULT_CONFIG.copy()

    elif algo == "A3C":
        agent = a3c.A3C
        config = a3c.DEFAULT_CONFIG.copy()

    elif algo == "PPO":
        agent = ppo.PPO
        config = ppo.DEFAULT_CONFIG.copy()

    elif algo == "SAC":
        agent = sac.SAC
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
        agent = ddpg.DDPG
        config = ddpg.DEFAULT_CONFIG.copy()

    elif algo == "APPO":
        agent = appo.APPO
        config = appo.DEFAULT_CONFIG.copy()

    elif algo == "IMPARA":
        agent = impala.Impala
        config = impala.DEFAULT_CONFIG.copy()

    else:
        raise ValueError
    return agent, config
