from typing import Dict, Optional

import numpy as np
import pandas as pd
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


class InvestmentCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs
    ) -> None:
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["equity"] = []
        # episode.hist_data["equity"] = []

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs
    ) -> None:
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        episode.user_data["equity"].append(worker.env.equity)

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs
    ) -> None:
        equity_curve = np.array(episode.user_data["equity"])
        returns = pd.Series(equity_curve).pct_change().dropna()
        cumulative_return = returns.sum()
        sharpe_ratio = returns.mean() / returns.std()
        episode.custom_metrics["cum_return"] = cumulative_return
        episode.custom_metrics["sharpe_ratio"] = sharpe_ratio
        # episode.hist_data["equity"] = episode.user_data["equity"]

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass
