from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from rl_bot.envs.environment import TradingEnv


class InformerScheme:
    def __init__(self, **kwargs) -> None:
        self.env: "TradingEnv" = None

    def reset(self, env: "TradingEnv") -> Dict[str, Any]:
        self.env = env
        return {}

    def step(self) -> Dict[str, Any]:
        return {}


class PrivateInformer(InformerScheme):
    def reset(self, env: "TradingEnv") -> Dict[str, Any]:
        return super().reset(env)

    def step(self) -> Dict[str, Any]:
        return super().step()


registry = {
    "PrivateInformer": PrivateInformer,
}


def get(class_name: str):
    if class_name not in list(registry.keys()):
        raise KeyError(f"{class_name} does not include the registry")
    return registry[class_name]
