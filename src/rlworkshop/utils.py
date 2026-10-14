class RewardTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.components = {}

    def add(self, name: str, value: float) -> float:
        self.components[name] = (
            self.components.get(name, 0.0) + value
        )
        return value

    def get_components(self):
        return self.components.copy()

    @property
    def total(self):
        return sum(self.components.values())


from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):

    def _on_step(self) -> bool:

        for info in self.locals["infos"]:

            components = info.get("reward_components")

            if components is None:
                continue

            for name, value in components.items():
                self.logger.record_mean(
                    f"reward_components/{name}",
                    value
                )

        return True