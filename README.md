# RLWorkshop

Lightweight, reusable utilities for reinforcement learning development and experimentation.

Version 0.1.0 implements **named reward tracking** in Gymnasium-compatible environments and **optional TensorBoard integration through Stable-Baselines3**.

## Installation

From the project directory:

```bash
python -m pip install -e .
```

For the Gymnasium example:

```bash
python -m pip install -e '.[gym]'
```

For SB3 and TensorBoard:

```bash
python -m pip install -e '.[sb3]'
```

To run tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

To build distributions for later publication or installation:

```bash
python -m build
```

Artifacts appear in `dist/`. This package has **not** been published to PyPI.

## Usage in a Gymnasium environment

```python
from rlworkshop import RewardTracker

# In __init__:
self.reward_tracker = RewardTracker(["step_penalty", "approach", "success"])

# In reset():
self.reward_tracker.reset()

# In step():
reward = 0.0
reward += self.reward_tracker.add("step_penalty", -0.01)
if closer_to_goal:
    reward += self.reward_tracker.add("approach", 0.1)
if success:
    reward += self.reward_tracker.add("success", 10.0)

# At the end of step(), after reward calculations:
info = {}
if terminated or truncated:
    info = self.reward_tracker.add_to_info(info)
return observation, reward, terminated, truncated, info
```

`RewardTracker.add()` returns a number so the original per-step `reward` is not changed beyond the intended reward components. The tracker separately accumulates totals and counts for the entire episode. `snapshot()` is immutable *with respect to subsequent tracker operations* (it returns detached ordinary dictionaries).

**Important:** include *every* reward and penalty contribution through `add()`. Otherwise the tracker total will not equal the actual episode return.

## SB3 integration

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from rlworkshop.callbacks import RewardComponentsCallback

# env = YourGymnasiumEnv()
env = Monitor(env)
model = PPO("MlpPolicy", env, tensorboard_log="./logs/")
model.learn(total_timesteps=100_000, callback=RewardComponentsCallback())
```

Start TensorBoard:

```bash
tensorboard --logdir ./logs/
```

Charts appear as `reward_components/<name>` and `reward_counts/<name>`. The
environment's total episodic return is available through SB3's standard
`rollout/ep_rew_mean` metric when using `Monitor`.

### Parallel environments and interpretation

- The environment sends final episode summaries in `info` when `terminated or truncated`.
- The callback implements only SB3's `_on_step()`: there is **no custom on_episode_end hook**.
- `logger.record_mean` aggregates across completed episodes until the next SB3 logger dump. Hence each point is a **mean per completed episode for that logging interval**, not a step-wise reward nor necessarily one point per episode.
- Declare component names in advance, including potentially unused components. Their totals and counts remain zero, preventing the accidental averaging of only episodes in which a component fired.
- Total reward components are *raw undiscounted episodic contributions*. Compare them with the environment's actual episodic return, not discounted training targets.
- When Gymnasium applies external `TimeLimit` truncation, a wrapper may end an episode outside the base environment. In that case use a tracker-aware wrapper or ensure the environment itself emits its final summary at truncation; otherwise the last summary may be absent.
- SB3's standard `rollout/ep_rew_mean` requires an episode-monitoring wrapper such as `Monitor`.

## Examples

```bash
python examples/lineworld.py
python examples/train_sb3.py
```

The core `RewardTracker` has no heavy dependencies; SB3 and TensorBoard are optional extras.
