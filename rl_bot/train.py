from typing import Any, Dict, Optional, Union

from ray import tune
from ray.rllib.agents import Trainer
from ray.tune import ExperimentAnalysis
from ray.tune.progress_reporter import CLIReporter, JupyterNotebookReporter
from ray.tune.stopper import Stopper


def train(
    agent_class: Trainer,
    config: Dict[str, Any],
    *,
    stop: Union[Dict[str, int], Stopper],
    expt_name: Optional[str] = None,
    num_samples: int = 1,
    local_dir: str = "./ray_results",
    resume: bool = False,
) -> ExperimentAnalysis:
    # try:
    #     class_name = get_ipython().__class__.__name__
    #     IS_NOTEBOOK = True if "Terminal" not in class_name else False
    # except NameError:
    #     IS_NOTEBOOK = False
    reporter = CLIReporter(
        metric_columns={
            "episode_reward_mean": "episode_reward",
            "evaluation/episode_reward_mean": "eval/episode_reward",
            "timesteps_total": "steps",
            "episodes_total": "episodes",
        },
        max_report_frequency=15,
    )
    # reporter = JupyterNotebookReporter(
    #     overwrite=True,
    #     metric_columns={
    #         "episode_reward_mean": "episode_reward",
    #         "evaluation/episode_reward_mean": "eval/episode_reward",
    #         "timesteps_total": "steps",
    #         "episodes_total": "episodes",
    #     },
    #     max_report_frequency=15,
    # )

    analysis = tune.run(
        agent_class,
        stop=stop,
        config=config,
        metric="episode_reward_mean",
        mode="max",
        checkpoint_at_end=True,
        progress_reporter=reporter,
        local_dir=local_dir,
        # verbose=1,
        name=expt_name,
        resume=resume,
        num_samples=num_samples,
        keep_checkpoints_num=2,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_freq=1,
    )

    # with open("analysis.pkl", "wb") as f:
    #     dill.dump(analysis, f)

    return analysis
