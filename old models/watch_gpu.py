import gymnasium as gym
from stable_baselines3 import PPO

# 1. Load the game WITH graphics so you can see it
# render_mode="human" forces the window to open
env = gym.make("CarRacing-v3", render_mode="human")

# 2. Load the brain you just trained
# make sure the name "my_gpu_racer" matches what you saved in train_gpu.py
print("Loading trained model...")
try:
    model = PPO.load("my_gpu_racer")
    print("Model loaded! Get ready to watch.")
except:
    print("Error: Could not find 'my_gpu_racer.zip'. Did the training finish?")
    exit()

# 3. The Loop: Let the AI drive
obs, _ = env.reset()
while True:
    # Ask the AI what to do based on what it sees
    action, _states = model.predict(obs)
    
    # Execute the action in the game
    obs, reward, terminated, truncated, info = env.step(action)
    
    # If the car crashes or finishes the lap, reset the track
    if terminated or truncated:
        obs, _ = env.reset()