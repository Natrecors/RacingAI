import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

if __name__ == "__main__":

    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    n_cpu = 4  

    print(f"Spinning up {n_cpu} parallel racing environments...")
    env = make_vec_env("CarRacing-v3", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)


    print("Loading AI Brain onto GTX 1080...")
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        device="cuda",       
        learning_rate=0.0003,
        n_steps=1024,       
        batch_size=256,      
        n_epochs=10,        
        ent_coef=0.01        
    )

    print("Starting High-Speed Training...")
    try:
        model.learn(total_timesteps=500000)
    except KeyboardInterrupt:
        print("Training stopped manually. Saving current progress...")


    model.save("my_gpu_racer")
    print("Training finished! Model saved as 'my_gpu_racer'.")
    env.close()