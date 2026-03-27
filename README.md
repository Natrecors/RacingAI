# RacingAI

# Autonomous Racing AI (CarRacing-v3)

This project trains an autonomous racing agent using Deep Reinforcement Learning. Using Proximal Policy Optimization (PPO) from `stable-baselines3`, the AI learns to drive around a procedurally generated track in the `CarRacing-v3` Gymnasium environment using pixel data (images) as input.


## 🚀 Features
* **Multiple Pre-trained Models:** Includes several generations of the AI, from a basic "Safe Racer" to a high-performing "Grandmaster Racer."
* **Optimized Training:** Custom wrappers (`MaxAndSkipEnv`, `GrayscaleObservation`, `ResizeObservation`) to process frames faster and stabilize training.
* **Hardware Support:** Scripts configured for both standard CPU training and high-speed GPU-accelerated training using PyTorch.

## 📁 Repository Structure
* `train_gpu.py` / `train_fast.py`: Scripts to train the PPO model from scratch or resume training.
* `train_stabilize.py`: Advanced training script utilizing frame stacking and multiprocessing (SubprocVecEnv) for stable learning.
* `watch_stable.py`: Run this script to watch the trained AI drive in real-time.
* `monitor.csv`: Exported telemetry and training statistics showing the model's reward improvement over time.
* `*.zip`: The saved PyTorch neural network weights for the different AI agents.
* `Graph.py`: Creating the Graph.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:Natrecors/RacingAI.git
   cd RacingAI

2. Install the required dependencies:
   ```bash
   pip install gymnasium[box2d] stable-baselines3[extra] torch torchvision matplotlib pandas
   
3. 🏎️ Usage
To watch the AI drive:
Make sure you have the required libraries installed, then simply run:
  ```bash
  python watch_stable.py
  ```
To train your own model:
If you have an NVIDIA GPU and want to train a new model from scratch:

    ```bash
    python train_gpu.py

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

Disclaimer: This is an educational AI project designed for the Gymnasium simulated environment.
