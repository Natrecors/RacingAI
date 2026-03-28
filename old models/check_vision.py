import gymnasium as gym
import numpy as np
import cv2
from gymnasium.wrappers import GrayscaleObservation

class CropDashboard(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
   
        new_shape = (84, old_shape[1], old_shape[2])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        return obs[:84, :, :]

# Create env
env = gym.make("CarRacing-v3", render_mode="rgb_array")
env = CropDashboard(env)
env = GrayscaleObservation(env, keep_dim=True)

print("Taking random steps to generate an image...")
obs, _ = env.reset()
for _ in range(50):
    obs, _, _, _, _ = env.step(env.action_space.sample())


img = obs.squeeze() 
img = np.array(img, dtype=np.uint8)

try:
    cv2.imshow("AI Vision Test", img)
    print("Look at the popup window. Is the road visible? Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("Could not open window (missing OpenCV).")