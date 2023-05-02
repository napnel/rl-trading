# from typing import Any, Dict

# import pandas as pd
# from ray.tune.registry import register_env
# from rl_trading.envs.components import action, informe, observe, reward, stop
# from rl_trading.envs.environment import TradingEnv

# # from components import action, informe, observe, reward, stop


# def create_env(env_config: Dict[str, Any]):
#     df_path = env_config.get("df_path")
#     df: pd.DataFrame = pd.read_pickle(df_path)
#     ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]
#     features_list = [name for name in list(df.columns) if name.islower()]
#     features = df[features_list]

#     observer = env_config.get("observer", "PublicObserver")
#     actions = env_config.get("actions", "MarketOrder")
#     rewards = env_config.get("rewards", "LogReturn")
#     informer = env_config.get("informer", "PriveteInformer")
#     stopper = env_config.get("stopper", "DrawdownStopper")
#     observer = observe.get(observer)
#     actions = action.get(actions)
#     rewards = reward.get(rewards)
#     informer = informe.get(informer)
#     stopper = stop.get(stopper)

#     components_config = {
#         "ohlcv": ohlcv,
#         "features": features,
#         "window_size": env_config["window_size"],
#     }
#     components_config.update(env_config.get("components_config", {}))

#     env = TradingEnv(
#         observer=observer(**components_config),
#         rewards=rewards(**components_config),
#         actions=actions(**components_config),
#         stopper=stopper(**components_config),
#         informer=informer(**components_config),
#         window_size=env_config["window_size"],
#         fee=env_config["fee"],
#     )
#     return env


# register_env("TradingEnv", create_env)
