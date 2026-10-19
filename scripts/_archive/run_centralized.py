import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add project root to path so we can run script directly from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN
from src.training.train_utils import train_epoch, evaluate
from src.utils.seed import set_seed
from src.utils.logger import CSVLogger, save_config
from src.utils.viz import plot_training_curves

def main():
    # Load config
    config_path = "configs/centralized_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config['paths']['results_dir'], f"run_{timestamp}")
    log_file = os.path.join(run_dir, "metrics.csv")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    figure_dir = os.path.join(run_dir, "figures")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    # Save a copy of the config used for this run
    save_config(config, os.path.join(run_dir, "config.json"))
    
    # Initialize logger
    logger = CSVLogger(log_file)
    
    # Set seed
    set_seed(config['seed'])
    
    # Set device
    device_pref = config['training']['device_preference']
    if device_pref == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # Data
    print("Setting up dataloaders...")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=config['data']['dataset_path'],
        batch_size=config['data']['batch_size']
    )
    
    # Model
    print("Initializing model...")
    model = SmallCNN().to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training Loop
    epochs = config['training']['epochs']
    best_acc = 0.0
    cumulative_time = 0.0
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        cumulative_time += epoch_time
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% - "
              f"Time: {epoch_time:.1f}s")
              
        # Log metrics
        logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'epoch_time': epoch_time,
            'cumulative_time': cumulative_time
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'config': config
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pt"))
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best.pt"))
            print(f"--> Saved new best model with acc {best_acc:.2f}%")
            
    print("Training complete. Generating plots...")
    plot_training_curves(log_file, figure_dir)
    print(f"Run outputs saved to: {run_dir}")

if __name__ == "__main__":
    main()
