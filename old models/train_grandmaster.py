import gymnasium as gym
import torch
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import GrayscaleObservation


N_CPU = 4                 
TOTAL_TIMESTEPS = 2000000 
LOG_DIR = "./logs_gm/"
MODEL_DIR = "./models_gm/"


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


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def make_env():
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    

    env = CropDashboard(env)
    

    env = GrayscaleObservation(env, keep_dim=True)
    
    return env

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn', force=True)


    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Initializing Grandmaster Training on {torch.cuda.get_device_name(0)}...")


    env = SubprocVecEnv([make_env for _ in range(N_CPU)])
    env = VecFrameStack(env, n_stack=4)
    

    env = VecMonitor(env, LOG_DIR)


    eval_callback = EvalCallback(
        env, 
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR, 
        eval_freq=25000, 
        deterministic=True, 
        render=False
    )


    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=linear_schedule(0.0003),
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )


    print("Starting Training... (Target: True Precision)")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    model.save("my_grandmaster_racer")
    print("Done! Saved final version.")
    env.close()