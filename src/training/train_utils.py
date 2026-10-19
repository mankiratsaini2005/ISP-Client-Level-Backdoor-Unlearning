import torch
import torch.nn as nn
import time
import numpy as np
from typing import Tuple, Dict, Any

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                device: torch.device) -> Tuple[float, float, float]:
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, epoch_time

def evaluate_extended(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
             criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
    """Evaluates the model and computes Acc, Loss, F1, Per-Class."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    num_classes = 10
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    # Calculate precision, recall, f1 per class
    per_class_metrics = {}
    macro_f1 = 0.0
    for i in range(num_classes):
        tp = confusion_matrix[i, i].item()
        fp = confusion_matrix[:, i].sum().item() - tp
        fn = confusion_matrix[i, :].sum().item() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        macro_f1 += f1
        per_class_metrics[f"class_{i}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    macro_f1 /= num_classes
    
    return {
        "loss": float(epoch_loss),
        "accuracy": float(epoch_acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist()
    }

def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
             criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    res = evaluate_extended(model, dataloader, criterion, device)
    return res["loss"], res["accuracy"]

