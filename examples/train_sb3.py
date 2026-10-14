"""Optional SB3 example: pip install -e '.[sb3]' then python examples/train_sb3.py"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from rlworkshop.callbacks import RewardComponentsCallback
from lineworld import LineWorld


def main():
    env = Monitor(LineWorld())
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", n_steps=128)
    model.learn(total_timesteps=10_000, callback=RewardComponentsCallback(), tb_log_name="lineworld")
    env.close()
    print("Run: tensorboard --logdir ./logs/")


if __name__ == "__main__":
    main()
