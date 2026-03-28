import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_path = os.path.join(current_dir, 'logs_stable', 'monitor.csv')
    
    print(f"Търся файл на адрес: {file_path}")

    if not os.path.exists(file_path):
        print(f"ГРЕШКА: Файлът не е намерен! Провери дали monitor.csv е в {file_path}")
        return

    print("Reading monitor.csv...")
    df = pd.read_csv(file_path, skiprows=1)

    df['rolling_reward'] = df['r'].rolling(window=50).mean()

    plt.figure(figsize=(10, 5))

    plt.plot(df.index, df['r'], alpha=0.3, color='tab:blue', label='Raw Episode Reward')
    plt.plot(df.index, df['rolling_reward'], color='tab:orange', linewidth=2, label='Rolling Average (50 episodes)')

    plt.title('AI Training Progress (CarRacing-v3)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Episodes', fontsize=12)
    plt.ylabel('Reward Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_path = os.path.join(current_dir, 'training_progress.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Graph saved as '{save_path}'.")
    
    plt.show()

if __name__ == "__main__":
    plot_results()
