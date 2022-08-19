import argparse
import json
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents import Trainer, ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.tune.logger import (CSVLogger, JsonLogger, TBXLogger,
                             TBXLoggerCallback, UnifiedLogger, pretty_print)
from ray.tune.registry import register_env
from ray.tune.stopper import MaximumIterationStopper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from ta.volatility import average_true_range

from rl_bot.backtest import backtest
from rl_bot.callbacks import InvestmentCallbacks
# from rl_bot.envs import create_env
from rl_bot.envs.environment import TradingEnv
from rl_bot.models.batch_norm import BatchNormModel
from rl_bot.models.rnn_network import RNNNetwork
from rl_bot.models.tcn import TCNNetwork
from rl_bot.train import train
from rl_bot.util import get_agent_class

# from tensortrade.env.default import create

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default="DQN", type=str)
parser.add_argument("--local_dir", default="./ray_results", type=str)
parser.add_argument("--expt-name", default=None, type=str)
parser.add_argument("--num_samples", default=1, type=int)
parser.add_argument("--num_cpus", default=os.cpu_count(), type=int)
parser.add_argument("--window_size", default=30, type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--cpt", default=None, type=str)
parser.add_argument("--pair", default="BTCUSDT", type=str)
args = parser.parse_args()

DATA_PATH = Path.home() / "data" / args.pair
TMP_PATH = Path("./tmp/").resolve()
EXPERIENCE_PATH = Path("./experience/").resolve()
CONFIG_PATH = Path("./config/").resolve()
LOG_PATH = Path("./log/").resolve()
DATA_PATH.mkdir(exist_ok=True, parents=True)
TMP_PATH.mkdir(exist_ok=True, parents=True)
EXPERIENCE_PATH.mkdir(exist_ok=True, parents=True)
CONFIG_PATH.mkdir(exist_ok=True, parents=True)
LOG_PATH.mkdir(exist_ok=True, parents=True)
ModelCatalog.register_custom_model("BatchNormModel", BatchNormModel)
ModelCatalog.register_custom_model("RNNNetwork", RNNNetwork)
ModelCatalog.register_custom_model("TCNNetwork", TCNNetwork)
tune.register_env("TradingEnv-v1", lambda config: TradingEnv(config))


def logger_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=[CSVLogger, JsonLogger, TBXLogger])

    return logger_creator


def test(agent: Trainer, env: TradingEnv):
    obs = env.reset()
    done = False
    while not done:
        action = agent.compute_single_action(obs, explore=False)
        obs, reward, done, info = env.step(action)
        # print(obs, reward, done, info)
    # print(agent.evaluate())
    # print(agent.evaluation_workers)
    # print(agent.workers)
    # print(agent.workers.local_worker())
    # print(agent.workers.local_worker().env)
    # print(agent.workers.local_worker().env.closed_trades)
    # print(wokers)
    # print(wokers.remote_workers())
    with open("./tmp/trades.pkl", "wb") as f:
        pickle.dump(env.trade_df, f)

    with open("./tmp/equity_curve.pkl", "wb") as f:
        pickle.dump(env.equity_curve_series, f)

    with open("./tmp/ohlcv.pkl", "wb") as f:
        pickle.dump(env.observer._ohlcv, f)


if __name__ == "__main__":
    ray.shutdown()
    ray.init(num_gpus=1, num_cpus=args.num_cpus)
    model_config = {
        # "use_attention": True,
        # "custom_model": tune.grid_search(["TCNNetwork", None]),
        # "custom_model": "TCNNetwork",
        # "custom_model_config": {
        #     "num_channels": [256, 128, 64, 16],
        # }
        # "dropout": 0.5,
        # },
        # "free_log_std": True,
    }

    agent_class, trainer_config = get_agent_class(args.algo)
    with open(CONFIG_PATH / "trainer.json", "r") as f:
        trainer_config = json.load(f)
        # trainer_config.update(config)

    print(trainer_config)
    # trainer_config["num_workers"] = args.num_cpus - 1
    # trainer_config["num_gpus"] = 0
    trainer_config["env_config"]["df_path"] = str(
        DATA_PATH / "features" / "df_4H_train.pkl"
    )
    trainer_config["evaluation_config"]["env_config"]["df_path"] = str(
        DATA_PATH / "features" / "df_4H_test.pkl"
    )
    trainer_config["model"] = model_config
    trainer_config["callbacks"] = InvestmentCallbacks
    trainer_config["num_workers"] = args.num_cpus - 1
    checkpoint_path = args.cpt

    if not args.test:

        agent = agent_class(
            config=trainer_config,
            logger_creator=logger_creator("./ray_results", args.algo),
        )
        for _ in range(30):
            results = agent.train()
            print(pretty_print(results))
            checkpoint_path = agent.save(f"./ray_results/{args.algo}")
            metrics = results["custom_metrics"]
            for key, values in metrics.items():
                if "mean" in key:
                    print(f"{key}: {values}")

        #     print("checkpoint:", checkpoint_path)
        # analysis = train(
        #     agent_class,
        #     trainer_config,
        #     stop={"timesteps_total": 1000000},
        #     expt_name=args.expt_name,
        #     num_samples=args.num_samples,
        #     local_dir=args.local_dir,
        #     resume=args.resume,
        # )
        # trial = analysis.get_best_trial()
        # checkpoint_path = analysis.get_best_checkpoint(trial)
        # trainer_config = analysis.get_best_config()
        # print(trainer_config)

    env_config_train = trainer_config["env_config"].copy()
    env_config_eval = trainer_config["evaluation_config"]["env_config"].copy()
    env_train = TradingEnv(env_config_train)
    env_test = TradingEnv(env_config_eval)

    trainer_config["logger_config"] = {"type": ray.tune.logger.NoopLogger}
    # trainer_config["evaluation_config"] = {}
    trainer_config["num_workers"] = 1
    agent = agent_class(config=trainer_config)
    if checkpoint_path:
        agent.restore(checkpoint_path)

    test(agent, env_test)
    backtest(env_train, agent, debug=False, plot=True, save_dir=str(TMP_PATH / "train"))
    backtest(env_test, agent, debug=False, plot=True, save_dir=str(TMP_PATH / "test"))

    ray.shutdown()
