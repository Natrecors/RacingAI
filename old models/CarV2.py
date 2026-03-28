import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gym.make("CarRacing-v3") 

if __name__ == "__main__":

    env = DummyVecEnv([make_env])

    model = PPO("CnnPolicy", env, verbose=1)

    print("Training started... (The window will NOT open, this is normal)")
    model.learn(total_timesteps=200000)

    # 4. Save
    model.save("my_fast_racing_ai")
    print("Training finished! Model saved.")