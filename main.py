# main.py
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from config import DEVICE, EPOCHS, BATCH_SIZE, PATIENCE, BASE_CONFIG
from dataset import get_data_loaders
from models import CustomMLP, CustomCNN
from engine import run_training_loop

def plot_ablation_results(all_results, metric_name="val_acc", title="Ablation Performance"):
    plt.figure(figsize=(10, 6))
    for run_name, history in all_results.items():
        plt.plot(history[metric_name], label=run_name)
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{metric_name}_ablation_benchmark.png")
    plt.show()

def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    # Let's perform a mini-ablation benchmarking project setup 
    # Dataset variations: 'mnist' and 'cifar10'
    # Architectural comparisons: CustomMLP and CustomCNN
    
    experiments = [
        {"dataset": "mnist", "arch": "MLP", "activation": "relu", "dropout": 0.0},
        {"dataset": "mnist", "arch": "CNN", "activation": "gelu", "dropout": 0.3},
        {"dataset": "cifar10", "arch": "CNN", "activation": "swish", "dropout": 0.3}
    ]
    
    all_runs_history = {}
    
    for idx, exp in enumerate(experiments):
        run_name = f"{exp['dataset']}_{exp['arch']}_{exp['activation']}_dr{exp['dropout']}"
        print(f"\n=== Launching Experiment {idx+1}/{len(experiments)}: {run_name} ===")
        
        # Load Data
        train_loader, val_loader, class_weights, in_channels, input_dim = get_data_loaders(
            dataset_name=exp['dataset'], batch_size=BATCH_SIZE
        )
        
        # Initialize Architecture
        if exp['arch'] == "MLP":
            model = CustomMLP(input_dim=input_dim, activation=exp['activation'], dropout=exp['dropout'])
        else:
            model = CustomCNN(in_channels=in_channels, activation=exp['activation'], dropout=exp['dropout'])
            
        model = model.to(DEVICE)
        
        # Print parameter scale
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters Count: {param_count:,}")
        
        # Loss and Optimizer setup
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=BASE_CONFIG['lr'], 
            weight_decay=BASE_CONFIG['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=EPOCHS, 
            eta_min=BASE_CONFIG['eta_min']
        )
        
        engine_config = {
            'epochs': EPOCHS,
            'patience': PATIENCE
        }
        
        checkpoint_file = f"checkpoints/{run_name}_best.pt"
        
        # Train
        history = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=engine_config,
            device=DEVICE,
            save_path=checkpoint_file
        )
        
        # Log final metrics metrics
        overfitting_gap = history['train_acc'][-1] - history['val_acc'][-1]
        print(f"-> Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
        print(f"-> Overfitting Gap (Train - Val): {overfitting_gap:.2f}%")
        print(f"-> Mean Inference Latency: {sum(history['latency'])/len(history['latency']):.4f} ms/img")
        
        all_runs_history[run_name] = history

    # Generate benchmark visualization plots
    plot_ablation_results(all_runs_history, "val_acc", "Validation Accuracy Across Variant Labs")
    plot_ablation_results(all_runs_history, "val_loss", "Validation Convergence Curves")

if __name__ == "__main__":
    main()