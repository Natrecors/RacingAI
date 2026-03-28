import gymnasium as gym
from stable_baselines3 import PPO


env = gym.make("CarRacing-v3", render_mode="human")


try:
    model = PPO.load("my_fast_racing_ai")
    print("Model loaded successfully!")
except:
    print("Could not find the model file yet. Did training finish?")
    exit()


obs, _ = env.reset()
while True:

    action, _states = model.predict(obs)
    

    obs, reward, terminated, truncated, info = env.step(action)
    

    if terminated or truncated:
        obs, _ = env.reset()