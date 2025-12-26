"""
Entraînement principal.

Usage:
    python -m src.train --config configs/config.yaml [--max_epochs 20] [--overfit_small]
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.model import build_model
from src.utils import set_seed, get_device, count_parameters, save_config_snapshot
from src.data_loading import get_dataloaders


def compute_accuracy(logits, labels):
    """Calcule l'accuracy."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pendant une époque."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, labels)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, val_loader, criterion, device):
    """Évalue le modèle sur la validation."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, labels)
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, filepath):
    """Sauvegarde un checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description="Entraînement complet")
    parser.add_argument('--config', type=str, required=True,
                       help='Chemin vers config.yaml')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed pour reproductibilité (défaut: depuis config)')
    parser.add_argument('--overfit_small', action='store_true',
                       help='Mode overfit sur petit échantillon')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Nombre max d\'époques (défaut: depuis config)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Nombre max de steps (défaut: depuis config)')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("ENTRAÎNEMENT COMPLET (Section 2.6)")
    print("="*60)
    
    # Seed
    seed = args.seed if args.seed is not None else config['train']['seed']
    set_seed(seed)
    print(f"\n✓ Seed fixée à {seed}")
    
    # Device
    device = get_device(config['train'].get('device', 'auto'))
    print(f"✓ Device: {device}")
    
    # Charger les données
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    num_classes = meta['num_classes']
    print(f"✓ DataLoaders chargés")
    print(f"  Nombre de classes: {num_classes}")
    
    # Mode overfit_small
    if args.overfit_small:
        print("\n⚠️  MODE OVERFIT SMALL ACTIVÉ")
        # Créer un petit DataLoader
        small_dataset = []
        count = 0
        num_samples = 32
        
        for images, labels in train_loader:
            for i in range(images.size(0)):
                if count >= num_samples:
                    break
                small_dataset.append((images[i], labels[i]))
                count += 1
            if count >= num_samples:
                break
        
        small_images = torch.stack([item[0] for item in small_dataset])
        small_labels = torch.stack([item[1] for item in small_dataset])
        small_tensor_dataset = torch.utils.data.TensorDataset(small_images, small_labels)
        train_loader = torch.utils.data.DataLoader(
            small_tensor_dataset,
            batch_size=min(len(small_dataset), config['train']['batch_size']),
            shuffle=True
        )
        print(f"  Train réduit à {len(train_loader.dataset)} exemples")
    
    # Construire le modèle
    print("\nConstruction du modèle...")
    model = build_model(config)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"✓ Modèle construit")
    print(f"  Paramètres: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Afficher les hyperparamètres
    model_config = config['model']
    print(f"\nHyperparamètres du modèle:")
    print(f"  - blocks_per_stage: {model_config['blocks_per_stage']}")
    print(f"  - dilation_stage3: {model_config['dilation_stage3']}")
    
    # Optimiseur
    optimizer_config = config['train']['optimizer']
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nOptimiseur:")
    print(f"  - Type: Adam")
    print(f"  - LR: {lr}")
    print(f"  - Weight decay: {weight_decay}")
    
    # Nombre d'époques
    epochs = args.max_epochs if args.max_epochs is not None else config['train']['epochs']
    print(f"\nEntraînement:")
    print(f"  - Époques: {epochs}")
    
    # TensorBoard
    runs_dir = config['paths']['runs_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_final_{timestamp}"
    log_dir = os.path.join(runs_dir, run_name)
    writer = SummaryWriter(log_dir)
    
    print(f"\nTensorBoard:")
    print(f"  - Log dir: {log_dir}")
    print(f"  - Tags: train/loss, val/loss, val/accuracy")
    
    # Sauvegarder la config
    save_config_snapshot(config, log_dir)
    
    # Checkpoint
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    best_ckpt_path = os.path.join(artifacts_dir, 'best.ckpt')
    
    # Stocker les valeurs pour les graphiques
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Meilleur modèle
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT ({epochs} époques)")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Logger dans TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        # Stocker pour graphiques
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Afficher le progrès
        print(f"Epoch {epoch+1:3d}/{epochs} | Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Sauvegarder le meilleur checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, epoch + 1, val_acc, val_loss, best_ckpt_path)
            print(f"  ✓ Nouveau meilleur modèle sauvegardé (Val Acc: {val_acc:.4f})")
        
        # Arrêt anticipé si max_steps spécifié
        if args.max_steps is not None:
            total_steps = (epoch + 1) * len(train_loader)
            if total_steps >= args.max_steps:
                print(f"\n✓ Max steps atteint ({args.max_steps}), arrêt anticipé")
                break
    
    writer.close()
    
    # Générer les graphiques
    print(f"\nGénération des graphiques...")
    
    # Graphique 1: Loss train vs val
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    plt.axvline(best_epoch, color='g', linestyle='--', linewidth=1, alpha=0.7, label=f'Meilleur (epoch {best_epoch})')
    plt.xlabel('Époque', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Évolution de la Loss (Train vs Validation)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Train Accuracy', marker='o', markersize=4)
    plt.plot(epochs_range, val_accs, 'r-', linewidth=2, label='Val Accuracy', marker='s', markersize=4)
    plt.axvline(best_epoch, color='g', linestyle='--', linewidth=1, alpha=0.7, label=f'Meilleur (epoch {best_epoch})')
    plt.xlabel('Époque', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Évolution de l\'Accuracy (Train vs Validation)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path1 = os.path.join(artifacts_dir, 'training_curves.png')
    plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique courbes: {plot_path1}")
    
    # Graphique 2: Comparaison train/val (zoom sur les dernières époques)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', linewidth=2, alpha=0.7, label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-', linewidth=2, alpha=0.7, label='Val Loss')
    plt.fill_between(epochs_range, train_losses, val_losses, alpha=0.2, color='gray', label='Écart train/val')
    plt.axvline(best_epoch, color='g', linestyle='--', linewidth=2, label=f'Meilleur (epoch {best_epoch})')
    plt.xlabel('Époque', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Train vs Validation (avec écart)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-', linewidth=2, alpha=0.7, label='Train Accuracy')
    plt.plot(epochs_range, val_accs, 'r-', linewidth=2, alpha=0.7, label='Val Accuracy')
    plt.fill_between(epochs_range, train_accs, val_accs, alpha=0.2, color='gray', label='Écart train/val')
    plt.axvline(best_epoch, color='g', linestyle='--', linewidth=2, label=f'Meilleur (epoch {best_epoch})')
    plt.xlabel('Époque', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Train vs Validation (avec écart)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path2 = os.path.join(artifacts_dir, 'training_curves_comparison.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique comparaison: {plot_path2}")
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"✓ Entraînement terminé")
    print(f"  - Époques: {epochs}")
    print(f"  - Meilleur modèle: Epoch {best_epoch}")
    print(f"  - Meilleure Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  - Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"  - Final Train Accuracy: {train_accs[-1]:.4f} ({train_accs[-1]*100:.2f}%)")
    print(f"  - Final Val Accuracy: {val_accs[-1]:.4f} ({val_accs[-1]*100:.2f}%)")
    print(f"\n✓ Configuration utilisée:")
    print(f"  - LR: {lr}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - dilation_stage3: {model_config['dilation_stage3']}")
    print(f"  - blocks_per_stage: {model_config['blocks_per_stage']}")
    print(f"\n✓ Checkpoint sauvegardé: {best_ckpt_path}")
    print(f"✓ Graphiques sauvegardés: {artifacts_dir}/training_curves*.png")
    print(f"✓ Logs TensorBoard: {log_dir}")


if __name__ == "__main__":
    main()
