import gymnasium as gym
import torch
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation


N_CPU = 4  
TOTAL_TIMESTEPS = 1000000 
LOG_DIR = "./logs_safe/"
MODEL_DIR = "./models_safe/"

def make_env():

    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    

    env = ResizeObservation(env, (64, 64))
    

    env = GrayscaleObservation(env, keep_dim=True)
    
    return env

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn', force=True)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Initializing SAFE MODE on {torch.cuda.get_device_name(0)}...")
    print("Cropping disabled. Vision is guaranteed.")


    env = SubprocVecEnv([make_env for _ in range(N_CPU)])
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env, LOG_DIR)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=512,         
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,          
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    print("Starting Training...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except KeyboardInterrupt:
        print("Interrupted. Saving...")

    model.save("my_safe_racer")
    print("Done. Saved as 'my_safe_racer.zip'")
    env.close()