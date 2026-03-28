import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

def make_env():
    env = gym.make("CarRacing-v3", render_mode="human")
    env = ResizeObservation(env, (64, 64)) # Match the Safe Mode size
    env = GrayscaleObservation(env, keep_dim=True)
    return env


env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

print("Loading Safe Mode Model...")
try:
    model = PPO.load("my_safe_racer", env=env)
    print("Model Loaded!")
except:
    print("Could not find 'my_safe_racer.zip'. Did you run train_safe.py?")
    exit()

# Run the car
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)