import gymnasium as gym
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv # <--- THE STABILIZER


N_CPU = 4
TOTAL_TIMESTEPS = 500000 
LOG_DIR = "./logs_stable/"
MODEL_DIR = "./models_stable/"

def make_env():
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    

    env = MaxAndSkipEnv(env, skip=2)
    
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)
    
    return env

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Initializing STABILIZER on {torch.cuda.get_device_name(0)}...")


    env = SubprocVecEnv([make_env for _ in range(N_CPU)])
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env, LOG_DIR)


    print("Loading your 913-score model...")
    try:

        model = PPO.load("my_safe_racer", env=env)
    except:
        print("Could not find 'my_safe_racer.zip'. Creating new one (Not recommended).")
        model = PPO("CnnPolicy", env, verbose=1, device="cuda")


    model.ent_coef = 0.01 
    model.learning_rate = 0.0001 

    print("Starting Stabilization Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save("my_stable_racer")
    print("Done! Saved as 'my_stable_racer.zip'")
    env.close()