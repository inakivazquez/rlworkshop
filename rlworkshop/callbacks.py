"""Optional Stable-Baselines3 integration; does not affect core imports."""

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:
    raise ImportError(
        "RewardComponentsCallback requires stable-baselines3. "
        "Install with: pip install 'rlworkshop[sb3]'"
    ) from exc


class RewardComponentsCallback(BaseCallback):
    """Record means across completed episodes in each SB3 log interval.

    The environment must put RewardTracker.snapshot() in the final step's
    ``info``. ``record_mean`` works with parallel vectorized environments.
    SB3 writes these scalar means at its normal logger dump intervals;
    an SB3 TensorBoard logger is required to view them in TensorBoard.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            components = info.get("reward_components")
            if components is None:
                continue
            counts = info.get("reward_counts", {})
            for name, value in components.items():
                self.logger.record_mean(f"reward_components/{name}", float(value))
            for name, value in counts.items():
                self.logger.record_mean(f"reward_counts/{name}", float(value))
            self.logger.record_mean("reward_components/total", sum(components.values()))
        return True
