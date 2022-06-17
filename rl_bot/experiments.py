import pathlib
from typing import Any, Dict

from ray import tune

from rl_bot.util import prepare_config_for_agent


class ExperimentCV(tune.Trainable):
    def setup(self, config: Dict[str, Any]):
        agent_class, algo_config = prepare_config_for_agent(config, self.logdir)
        algo_config.pop("_env_test_config")  # When training, test data is not used.
        self.agent = agent_class(config=algo_config)

    def step(self):
        return self.agent.train()

    def save_checkpoint(self, checkpoint_dir: str):
        return self.agent.save(pathlib.Path(checkpoint_dir).parent)
