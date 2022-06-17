import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ray
from ray.tune import ExperimentAnalysis

from rl_bot.backtest import backtest
from rl_bot.envs import TradingEnv
from rl_bot.tuning_space import get_tuning_params
from rl_bot.util import prepare_config_for_agent


def get_expt_results_cv(analysis: ExperimentAnalysis):
    all_configs = analysis.get_all_configs()
    expt_df = pd.DataFrame()
    parameter_table = pd.DataFrame()

    for logdir, df in analysis.trial_dataframes.items():
        df = df[
            ["episode_reward_mean", "evaluation/episode_reward_mean", "timesteps_total"]
        ]
        config = all_configs[logdir].copy()
        for k, v in config["env_config"].items():
            config[f"env_config/{k}"] = v
        config.pop("env_config")
        tuned_params = list(get_tuning_params(config["_algo"]).keys())
        tuned_config = {k: v for k, v in config.items() if k in tuned_params}
        tuned_config = pd.DataFrame(tuned_config, index=[0])
        parameter_table = pd.concat([parameter_table, tuned_config], axis=0)

        df = pd.concat([df, tuned_config], axis=1).fillna(method="ffill")
        expt_df = pd.concat([expt_df, df], axis=0)

    parameter_table = parameter_table.drop_duplicates()
    parameter_table.index = range(0, len(parameter_table.index))
    results_cv = expt_df.groupby(by=tuned_params + ["timesteps_total"]).mean()

    save_dir = Path(logdir).parent.joinpath("results.pkl")
    with open(save_dir, "wb") as f:
        pickle.dump(results_cv, f)
    return results_cv


def get_best_expt(analysis: ExperimentAnalysis):
    results_cv = get_expt_results_cv(analysis)
    tuned_params = list(get_tuning_params(analysis.best_config["_algo"]).keys())
    # print(results_cv["evaluation/episode_reward_mean"].rolling(5).mean())
    last_5_avg: pd.DataFrame = (
        results_cv["evaluation/episode_reward_mean"]
        .rolling(5)
        .mean()
        .groupby(tuned_params)
        .last()
        .sort_values(ascending=False)
    )
    best_config = dict(zip(last_5_avg.index.names, last_5_avg.index[0]))
    best_progress = results_cv.loc[tuple(best_config.values()), :]
    print(best_config)
    print(best_progress)
    return best_progress, best_config


# Learning Curve
def plot_all_progress_cv(analysis: ExperimentAnalysis):
    expt_results_cv = get_expt_results_cv(analysis)
    tuned_params = list(get_tuning_params(analysis.best_config["_algo"]).keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    periods = 5
    for name, group in expt_results_cv.groupby(tuned_params):
        x = group.index.get_level_values(-1)
        reward_mean_train = group["episode_reward_mean"].rolling(periods).mean().values
        reward_mean_eval = (
            group["evaluation/episode_reward_mean"].rolling(periods).mean().values
        )
        axes[0].plot(x, reward_mean_train, label=name)
        axes[1].plot(x, reward_mean_eval)

    axes[0].grid()
    axes[1].grid()


def plot_best_progress_cv(analysis: ExperimentAnalysis):
    expt_results_cv = get_expt_results_cv(analysis)
    tuned_params = list(get_tuning_params(analysis.best_config["_algo"]).keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    periods = 5
    for name, group in expt_results_cv.groupby(tuned_params):
        x = group.index.get_level_values(-1)
        reward_mean_train = group["episode_reward_mean"].rolling(periods).mean().values
        reward_mean_eval = (
            group["evaluation/episode_reward_mean"].rolling(periods).mean().values
        )
        axes[0].plot(x, reward_mean_train, label=name)
        axes[1].plot(x, reward_mean_eval)

    axes[0].grid()
    axes[1].grid()


def get_best_trials(analysis: ExperimentAnalysis, best_config: dict):
    all_configs = analysis.get_all_configs()
    best_trial_ids = []
    for logdir, df in analysis.trial_dataframes.items():
        config = all_configs[logdir].copy()
        is_match = True
        for key, value in best_config.items():
            if key == "env_config/window_size":
                continue
            if config[key] != value:
                is_match = False

        if is_match:
            best_trial_ids.append(df["trial_id"][0])

    best_trials = []
    for trial in analysis.trials:
        if trial.trial_id in best_trial_ids:
            best_trials.append(trial)

    print(f"Best Trials: {best_trials}")
    return best_trials


def backtest_expt(analysis: ExperimentAnalysis, debug=False):
    all_config = analysis.get_all_configs()
    best_progress, best_config = get_best_expt(analysis)
    best_trials = get_best_trials(analysis, best_config)

    agent = None
    for trial in best_trials:
        config = all_config[trial.logdir].copy()
        fold_id = config["__trial_index__"]
        agent_class, algo_config = prepare_config_for_agent(config, Path(trial.logdir))
        env_test_config = algo_config.pop("_env_test_config")
        algo_config["num_workers"] = 1
        algo_config["logger_config"] = {"type": ray.tune.logger.NoopLogger}

        if agent is None:
            agent = agent_class(config=algo_config)
        else:
            agent.setup(algo_config)

        checkpoint = analysis.get_best_checkpoint(trial)
        agent.restore(checkpoint)

        env_train = TradingEnv(**algo_config["env_config"])
        env_eval = TradingEnv(**algo_config["evaluation_config"]["env_config"])
        env_test = TradingEnv(**env_test_config)

        backtest_dir = Path(trial.logdir).resolve().parent / "backtest-stats"

        backtest(
            env_train,
            agent,
            save_dir=os.path.join(backtest_dir, f"train-{fold_id}"),
            plot=True,
            open_browser=debug,
        )
        backtest(
            env_eval,
            agent,
            save_dir=os.path.join(backtest_dir, f"eval-{fold_id}"),
            plot=True,
            open_browser=debug,
        )
        backtest(
            env_test,
            agent,
            save_dir=os.path.join(backtest_dir, f"test-{fold_id}"),
            plot=True,
            open_browser=debug,
        )

        backtest(
            env_train,
            agent="Buy&Hold",
            save_dir=os.path.join(
                backtest_dir.parent.parent,
                "backtest-stats-buy&hold",
                f"train-{fold_id}",
            ),
            plot=False,
        )
        backtest(
            env_eval,
            agent="Buy&Hold",
            save_dir=os.path.join(
                backtest_dir.parent.parent, "backtest-stats-buy&hold", f"eval-{fold_id}"
            ),
            plot=False,
        )
        backtest(
            env_test,
            agent="Buy&Hold",
            save_dir=os.path.join(
                backtest_dir.parent.parent, "backtest-stats-buy&hold", f"test-{fold_id}"
            ),
            plot=False,
        )
