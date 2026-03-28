import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import GrayscaleObservation


class CropDashboard(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        new_shape = (84, old_shape[1], old_shape[2])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs):
        return obs[:84, :, :]

def make_env():
    env = gym.make("CarRacing-v3", render_mode="human")
    env = CropDashboard(env)
    env = GrayscaleObservation(env, keep_dim=True)
    return env


env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)


try:
    print("Loading Best Model...")
    model = PPO.load("models_gm/best_model", env=env)
except:
    print("Loading Final Model...")
    model = PPO.load("my_grandmaster_racer", env=env)


obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)