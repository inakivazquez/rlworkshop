"""Minimal standalone Gymnasium example: python examples/lineworld.py"""

import gymnasium as gym
from gymnasium import spaces
from rlworkshop import RewardTracker


class LineWorld(gym.Env):
    """Move from position 0 to 4; reach position 4 to succeed."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)  # 0: left; 1: right
        self.reward_tracker = RewardTracker(
            ["step_penalty", "approach", "success"]
        )
        self.position = 0
        self.n_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.n_steps = 0
        self.reward_tracker.reset()
        return self.position, {}

    def step(self, action):
        old_position = self.position
        self.position = min(4, max(0, self.position + (1 if action == 1 else -1)))
        self.n_steps += 1
        reward = self.reward_tracker.add("step_penalty", -0.01)
        if self.position > old_position:
            reward += self.reward_tracker.add("approach", 0.1)
        terminated = self.position == 4
        truncated = self.n_steps >= 20 and not terminated
        if terminated:
            reward += self.reward_tracker.add("success", 1.0)
        info = {}
        if terminated or truncated:
            info = self.reward_tracker.add_to_info(info)
        return self.position, reward, terminated, truncated, info


if __name__ == "__main__":
    env = LineWorld()
    obs, info = env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        print(f"state={obs}, reward={reward:.2f}")
    print("episode summary:", info)
