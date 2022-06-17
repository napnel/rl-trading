import time


class StopperScheme:
    def __init__(self, **kwargs) -> None:
        self.env = None

    def reset(self, env) -> bool:
        self.env = env
        self.max_step = len(self.env.observer.ohlcv)
        return False

    def step(self) -> bool:
        done = self.env.current_step >= self.max_step - 1
        return done


class DrawdownStopper(StopperScheme):
    def __init__(self, allowable_drawdown: float = 0.5, **kwargs) -> None:
        super().__init__()
        assert 0 <= allowable_drawdown < 1
        self.allowable_drawdown = allowable_drawdown

    def reset(self, env) -> bool:
        super().reset(env)
        self.initial_cash = self.env.initial_cash

    def step(self) -> bool:
        done = super().step()
        return done or self.env.equity < (self.initial_cash * self.allowable_drawdown)


registry = {
    "DrawdownStopper": DrawdownStopper,
}


def get(class_name: str):
    if class_name not in list(registry.keys()):
        raise KeyError(f"{class_name} does not include the registry")
    return registry[class_name]
