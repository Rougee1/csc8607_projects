"""
Overfit sur un très petit échantillon (section 2.3).

Usage:
    python -m src.overfit_small --config configs/config.yaml --overfit_size 32
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
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import matplotlib.pyplot as plt
import numpy as np
from src.model import build_model
from src.utils import set_seed, get_device, count_parameters, save_config_snapshot
from src.data_loading import get_dataloaders


def create_small_dataloader(dataloader, num_samples):
    """
    Crée un DataLoader tronqué avec seulement num_samples exemples.
    
    Args:
        dataloader: DataLoader original
        num_samples: Nombre d'exemples à garder
    
    Returns:
        Nouveau DataLoader avec seulement num_samples exemples
    """
    # Collecter les premiers num_samples exemples
    small_dataset = []
    count = 0
    
    for images, labels in dataloader:
        for i in range(images.size(0)):
            if count >= num_samples:
                break
            small_dataset.append((images[i], labels[i]))
            count += 1
        if count >= num_samples:
            break
    
    # Créer un nouveau DataLoader
    small_images = torch.stack([item[0] for item in small_dataset])
    small_labels = torch.stack([item[1] for item in small_dataset])
    small_tensor_dataset = torch.utils.data.TensorDataset(small_images, small_labels)
    
    return torch.utils.data.DataLoader(
        small_tensor_dataset,
        batch_size=min(len(small_dataset), dataloader.batch_size),
        shuffle=True
    )


def main():
    parser = argparse.ArgumentParser(description="Overfit sur un petit échantillon")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers config.yaml')
    parser.add_argument('--overfit_size', type=int, default=32,
                       help='Nombre d\'exemples pour l\'overfit (16-64 recommandé)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'époques')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (plus élevé que normal pour overfit)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour reproductibilité')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("OVERFIT SUR PETIT ÉCHANTILLON (Section 2.3)")
    print("="*60)
    
    # Fixer la seed
    set_seed(args.seed)
    print(f"\n✓ Seed fixée à {args.seed}")
    
    # Détecter le device
    device = get_device(config['train'].get('device', 'auto'))
    print(f"✓ Device: {device}")
    
    # Charger les données
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    num_classes = meta['num_classes']
    print(f"✓ DataLoaders chargés")
    print(f"  Nombre de classes: {num_classes}")
    
    # Créer un petit DataLoader
    print(f"\nCréation d'un petit DataLoader avec {args.overfit_size} exemples...")
    small_train_loader = create_small_dataloader(train_loader, args.overfit_size)
    print(f"✓ Petit DataLoader créé: {len(small_train_loader.dataset)} exemples")
    print(f"  Batch size: {small_train_loader.batch_size}")
    
    # Construire le modèle
    print("\nConstruction du modèle...")
    model = build_model(config)
    model = model.to(device)
    model.train()
    
    num_params = count_parameters(model)
    print(f"✓ Modèle construit")
    print(f"  Paramètres: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Afficher les hyperparamètres du modèle
    model_config = config['model']
    print(f"\nHyperparamètres du modèle:")
    print(f"  - blocks_per_stage: {model_config['blocks_per_stage']}")
    print(f"  - dilation_stage3: {model_config['dilation_stage3']}")
    
    # Optimiseur
    optimizer_config = config['train']['optimizer']
    weight_decay = args.lr * 0.01 if optimizer_config.get('weight_decay', 0) > 0 else 0.0
    # Pour overfit, utiliser un LR plus élevé et weight_decay faible
    lr = args.lr
    weight_decay = 0.0  # Pas de weight decay pour overfit
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nOptimiseur:")
    print(f"  - Type: Adam")
    print(f"  - LR: {lr}")
    print(f"  - Weight decay: {weight_decay}")
    
    # TensorBoard
    runs_dir = config['paths']['runs_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"overfit_small_{args.overfit_size}ex_{timestamp}"
    log_dir = os.path.join(runs_dir, run_name)
    writer = SummaryWriter(log_dir)
    
    print(f"\nTensorBoard:")
    print(f"  - Log dir: {log_dir}")
    print(f"  - Tag: train/loss")
    
    # Sauvegarder la config
    save_config_snapshot(config, log_dir)
    
    # Entraînement
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT ({args.epochs} époques)")
    print(f"{'='*60}")
    
    # Stocker les valeurs de loss pour le graphique
    epoch_losses = []
    iteration_losses = []
    iteration_steps = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(small_train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Logger à chaque itération
            global_step = epoch * len(small_train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            
            # Stocker pour le graphique
            iteration_losses.append(loss.item())
            iteration_steps.append(global_step)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Stocker la loss moyenne par époque
        epoch_losses.append(avg_loss)
        
        # Logger aussi par époque
        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        
        # Afficher le progrès
        if (epoch + 1) % 5 == 0 or epoch == 0 or avg_loss < 0.1:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.6f}")
        
        # Arrêter si la loss est très faible
        if avg_loss < 0.01:
            print(f"\n✓ Loss très faible ({avg_loss:.6f}), arrêt anticipé")
            break
    
    writer.close()
    
    # Créer le graphique
    print(f"\nGénération du graphique...")
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Graphique 1: Loss par époque
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(epoch_losses) + 1)
    plt.plot(epochs_range, epoch_losses, 'b-o', linewidth=2, markersize=6, label='Loss moyenne par époque')
    plt.xlabel('Époque', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Overfit Small - Loss d\'entraînement\n({args.overfit_size} exemples, LR={lr}, blocks={model_config["blocks_per_stage"]}, dilation={model_config["dilation_stage3"]})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.yscale('log')  # Échelle logarithmique pour mieux voir la descente
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plot_filename = f"overfit_small_loss_{args.overfit_size}ex.png"
    plot_path = os.path.join(artifacts_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé: {plot_path}")
    
    # Graphique 2: Loss par itération (plus détaillé)
    if len(iteration_losses) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(iteration_steps, iteration_losses, 'b-', linewidth=1.5, alpha=0.7, label='Loss par itération')
        plt.xlabel('Itération', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Overfit Small - Loss d\'entraînement (détaillé)\n({args.overfit_size} exemples, {len(epoch_losses)} époques)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.yscale('log')
        plt.tight_layout()
        
        plot_filename_detailed = f"overfit_small_loss_detailed_{args.overfit_size}ex.png"
        plot_path_detailed = os.path.join(artifacts_dir, plot_filename_detailed)
        plt.savefig(plot_path_detailed, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Graphique détaillé sauvegardé: {plot_path_detailed}")
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"✓ Overfit terminé")
    print(f"  - Taille échantillon: {args.overfit_size} exemples")
    print(f"  - Époques: {epoch+1}")
    print(f"  - Loss finale: {avg_loss:.6f}")
    print(f"  - Hyperparamètres modèle: blocks_per_stage={model_config['blocks_per_stage']}, dilation_stage3={model_config['dilation_stage3']}")
    print(f"  - LR: {lr}, Weight decay: {weight_decay}")
    print(f"\n✓ Logs TensorBoard sauvegardés dans: {log_dir}")
    print(f"  Visualiser avec: tensorboard --logdir {log_dir}")
    print(f"\n✓ Graphiques sauvegardés dans: {artifacts_dir}")
    print(f"  - {plot_filename} (loss par époque)")
    if len(iteration_losses) > 0:
        print(f"  - {plot_filename_detailed} (loss par itération)")
    print(f"\n✓ Graphique prêt pour le rapport (section M3)")


if __name__ == "__main__":
    main()

