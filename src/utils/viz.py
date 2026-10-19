import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_curves(log_csv_path: str, save_dir: str):
    """
    Reads the CSV log and generates required plots:
    1. Training loss vs epoch
    2. Test loss vs epoch
    3. Training accuracy vs epoch
    4. Test accuracy vs epoch
    """
    if not os.path.exists(log_csv_path):
        print(f"Log file {log_csv_path} not found. Cannot plot curves.")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(log_csv_path)

    epochs = df['epoch']
    
    # 1. Training loss vs epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_loss_vs_epoch.png'))
    plt.close()

    # 2. Test loss vs epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['test_loss'], label='Test Loss', color='red', marker='o')
    plt.title('Test Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_loss_vs_epoch.png'))
    plt.close()

    # 3. Training accuracy vs epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['train_acc'], label='Train Accuracy', color='green', marker='o')
    plt.title('Training Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_accuracy_vs_epoch.png'))
    plt.close()

    # 4. Test accuracy vs epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, df['test_acc'], label='Test Accuracy', color='orange', marker='o')
    plt.title('Test Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_accuracy_vs_epoch.png'))
    plt.close()
