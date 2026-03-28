import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

def make_env():
    env = gym.make("CarRacing-v3", render_mode="human")
    env = MaxAndSkipEnv(env, skip=2) 
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

print("Loading Stable Model...")
try:
    model = PPO.load("my_stable_racer", env=env)
except:
    print("Stable model not found, trying safe racer...")
    model = PPO.load("my_safe_racer", env=env)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)