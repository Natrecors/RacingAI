import gymnasium as gym
from stable_baselines3 import PPO


env = gym.make("CarRacing-v3", render_mode="human")


model = PPO("CnnPolicy", env, verbose=1)


print("Starting training... The window might freeze for a second while loading.")
model.learn(total_timesteps=100000)


model.save("my_racing_ai")
print("Training finished!")