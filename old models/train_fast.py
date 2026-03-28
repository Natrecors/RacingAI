import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env():
    return gym.make("CarRacing-v3")

if __name__ == "__main__":
    env = DummyVecEnv([make_env])


    print("Loading previous brain...")
    model = PPO.load("my_fast_racing_ai", env=env)


    print("Resuming training...")
    model.learn(total_timesteps=100000)


    model.save("my_fast_racing_ai")
    print("Training finished!")