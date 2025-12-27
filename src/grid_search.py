"""
Mini grid search — recherche d'hyperparamètres.

Usage:
    python -m src.grid_search --config configs/config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import itertools
import json
import pandas as pd
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


def main():
    parser = argparse.ArgumentParser(description="Mini grid search")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers config.yaml')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Nombre d\'époques par run (défaut: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour reproductibilité')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("MINI GRID SEARCH (Section 2.5)")
    print("="*60)
    
    # Fixer la seed globale
    set_seed(args.seed)
    print(f"\n✓ Seed globale fixée à {args.seed} (identique pour tous les runs)")
    
    # Détecter le device
    device = get_device(config['train'].get('device', 'auto'))
    print(f"✓ Device: {device}")
    
    # Charger les données
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    num_classes = meta['num_classes']
    print(f"✓ DataLoaders chargés")
    print(f"  Nombre de classes: {num_classes}")
    
    # Lire les hyperparamètres de la grille
    hparams = config['hparams']
    lrs = hparams['lr']
    # Convertir weight_decay en float si ce sont des strings
    weight_decays_raw = hparams['weight_decay']
    weight_decays = [float(wd) if isinstance(wd, str) else wd for wd in weight_decays_raw]
    dilation_stage3_values = hparams['dilation_stage3']
    blocks_per_stage_values = hparams['blocks_per_stage']
    
    print(f"\nGrille d'hyperparamètres:")
    print(f"  - LR: {lrs}")
    print(f"  - Weight decay: {weight_decays}")
    print(f"  - dilation_stage3: {dilation_stage3_values}")
    print(f"  - blocks_per_stage: {blocks_per_stage_values}")
    
    total_combinations = len(lrs) * len(weight_decays) * len(dilation_stage3_values) * len(blocks_per_stage_values)
    print(f"  - Total: {total_combinations} combinaisons")
    print(f"  - Époques par run: {args.epochs}")
    
    # Créer toutes les combinaisons
    combinations = list(itertools.product(lrs, weight_decays, dilation_stage3_values, blocks_per_stage_values))
    
    # Stocker les résultats
    results = []
    runs_dir = config['paths']['runs_dir']
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Fichier de checkpoint pour reprendre en cas d'interruption
    checkpoint_file = os.path.join(artifacts_dir, 'grid_search_checkpoint.json')
    
    # Charger les résultats existants si le fichier existe
    import json
    completed_runs = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                completed_runs = set(checkpoint_data.get('completed_runs', []))
            print(f"\n✓ Checkpoint trouvé: {len(results)} runs déjà complétés")
            print(f"  Reprise depuis le run {len(results)+1}/{total_combinations}")
        except:
            print(f"\n⚠️  Checkpoint corrompu, redémarrage depuis le début")
    
    print(f"\n{'='*60}")
    print(f"EXÉCUTION DE LA GRID SEARCH ({total_combinations} runs)")
    print(f"{'='*60}")
    
    for run_idx, (lr, weight_decay, dilation_stage3, blocks_per_stage) in enumerate(combinations):
        # S'assurer que weight_decay est un float
        weight_decay_float = float(weight_decay) if isinstance(weight_decay, str) else weight_decay
        
        # Nom du run
        run_name = f"grid_lr={lr:.4f}_wd={weight_decay_float:.0e}_dil={dilation_stage3}_blk={blocks_per_stage}"
        
        # Vérifier si ce run a déjà été complété
        if run_name in completed_runs:
            print(f"\n[{run_idx+1}/{total_combinations}] {run_name} - DÉJÀ COMPLÉTÉ (skip)")
            continue
        
        print(f"\n[{run_idx+1}/{total_combinations}] {run_name}")
        print(f"  LR={lr:.6f}, WD={weight_decay_float:.0e}, Dilation={dilation_stage3}, Blocks={blocks_per_stage}")
        
        # Créer un sous-config pour ce run
        run_config = config.copy()
        run_config['model']['dilation_stage3'] = dilation_stage3
        run_config['model']['blocks_per_stage'] = blocks_per_stage
        run_config['train']['optimizer']['lr'] = lr
        run_config['train']['optimizer']['weight_decay'] = weight_decay_float
        
        # Fixer la seed pour ce run (même seed pour tous)
        set_seed(args.seed)
        
        # Construire le modèle avec ces hyperparamètres
        model = build_model(run_config)
        model = model.to(device)
        
        # Optimiseur
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_float)
        criterion = nn.CrossEntropyLoss()
        
        # TensorBoard pour ce run
        log_dir = os.path.join(runs_dir, run_name)
        writer = SummaryWriter(log_dir)
        
        # Sauvegarder la config pour ce run
        save_config_snapshot(run_config, log_dir)
        
        # Logger les hyperparamètres dans TensorBoard HParams
        writer.add_hparams(
            {
                'lr': lr,
                'weight_decay': weight_decay_float,
                'dilation_stage3': dilation_stage3,
                'blocks_per_stage': blocks_per_stage,
            },
            {}  # Métriques seront ajoutées après
        )
        
        # Entraînement
        best_val_acc = 0.0
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validation
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            # Logger
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/accuracy', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/accuracy', val_acc, epoch)
            
            # Mettre à jour le meilleur
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
            
            if epoch == args.epochs - 1:
                print(f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        writer.close()
        
        # Stocker les résultats
        results.append({
            'run_name': run_name,
            'lr': lr,
            'weight_decay': weight_decay_float,
            'dilation_stage3': dilation_stage3,
            'blocks_per_stage': blocks_per_stage,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
        })
        
        # Sauvegarder le checkpoint après chaque run
        completed_runs.add(run_name)
        checkpoint_data = {
            'results': results,
            'completed_runs': list(completed_runs)
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"    ✓ Checkpoint sauvegardé ({len(results)}/{total_combinations} runs)")
    
    # Créer un DataFrame pour le tableau
    df = pd.DataFrame(results)
    
    # Trier par meilleure val_acc
    df_sorted = df.sort_values('best_val_acc', ascending=False)
    
    print(f"\n{'='*60}")
    print("RÉSULTATS - TABLEAU RÉCAPITULATIF")
    print(f"{'='*60}")
    print(df_sorted[['lr', 'weight_decay', 'dilation_stage3', 'blocks_per_stage', 'best_val_acc', 'best_val_loss']].to_string(index=False))
    
    # Sauvegarder le tableau en CSV
    csv_path = os.path.join(artifacts_dir, 'grid_search_results.csv')
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n✓ Tableau sauvegardé: {csv_path}")
    
    # Meilleure combinaison
    best_run = df_sorted.iloc[0]
    print(f"\n{'='*60}")
    print("MEILLEURE COMBINAISON")
    print(f"{'='*60}")
    print(f"  LR: {best_run['lr']:.6f}")
    print(f"  Weight decay: {best_run['weight_decay']:.0e}")
    print(f"  dilation_stage3: {best_run['dilation_stage3']}")
    print(f"  blocks_per_stage: {best_run['blocks_per_stage']}")
    print(f"  Val Accuracy: {best_run['best_val_acc']:.4f}")
    print(f"  Val Loss: {best_run['best_val_loss']:.4f}")
    
    # Générer des graphiques
    print(f"\nGénération des graphiques...")
    
    # Graphique 1: Heatmap Val Accuracy par hyperparamètres
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Heatmap: LR vs Weight Decay (moyenne sur dilation et blocks)
    pivot_lr_wd = df.groupby(['lr', 'weight_decay'])['best_val_acc'].mean().unstack()
    im1 = axes[0, 0].imshow(pivot_lr_wd.values, aspect='auto', cmap='viridis')
    axes[0, 0].set_xticks(range(len(pivot_lr_wd.columns)))
    axes[0, 0].set_xticklabels([f'{wd:.0e}' for wd in pivot_lr_wd.columns])
    axes[0, 0].set_yticks(range(len(pivot_lr_wd.index)))
    axes[0, 0].set_yticklabels([f'{lr:.4f}' for lr in pivot_lr_wd.index])
    axes[0, 0].set_xlabel('Weight Decay', fontsize=12)
    axes[0, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[0, 0].set_title('Val Accuracy: LR vs Weight Decay', fontsize=14)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Heatmap: Dilation vs Blocks (moyenne sur LR et WD)
    pivot_dil_blk = df.groupby(['dilation_stage3', 'blocks_per_stage'])['best_val_acc'].mean().unstack()
    im2 = axes[0, 1].imshow(pivot_dil_blk.values, aspect='auto', cmap='viridis')
    axes[0, 1].set_xticks(range(len(pivot_dil_blk.columns)))
    axes[0, 1].set_xticklabels(pivot_dil_blk.columns)
    axes[0, 1].set_yticks(range(len(pivot_dil_blk.index)))
    axes[0, 1].set_yticklabels(pivot_dil_blk.index)
    axes[0, 1].set_xlabel('Blocks per Stage', fontsize=12)
    axes[0, 1].set_ylabel('Dilation Stage 3', fontsize=12)
    axes[0, 1].set_title('Val Accuracy: Dilation vs Blocks', fontsize=14)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Graphique: Val Accuracy par LR
    axes[1, 0].boxplot([df[df['lr'] == lr]['best_val_acc'].values for lr in sorted(df['lr'].unique())], 
                       tick_labels=[f'{lr:.4f}' for lr in sorted(df['lr'].unique())])
    axes[1, 0].set_xlabel('Learning Rate', fontsize=12)
    axes[1, 0].set_ylabel('Val Accuracy', fontsize=12)
    axes[1, 0].set_title('Distribution Val Accuracy par LR', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Graphique: Val Accuracy par hyperparamètres du modèle
    dilation_acc = [df[df['dilation_stage3'] == d]['best_val_acc'].values for d in sorted(df['dilation_stage3'].unique())]
    blocks_acc = [df[df['blocks_per_stage'] == b]['best_val_acc'].values for b in sorted(df['blocks_per_stage'].unique())]
    
    x_pos = np.arange(2)
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, [np.mean(d) for d in dilation_acc], width, label='Dilation', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, [np.mean(b) for b in blocks_acc], width, label='Blocks', alpha=0.7)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['2', '3'])
    axes[1, 1].set_xlabel('Valeur', fontsize=12)
    axes[1, 1].set_ylabel('Val Accuracy (moyenne)', fontsize=12)
    axes[1, 1].set_title('Val Accuracy par Hyperparamètre Modèle', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path1 = os.path.join(artifacts_dir, 'grid_search_analysis.png')
    plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique d'analyse: {plot_path1}")
    
    # Graphique 2: Comparaison des meilleures combinaisons
    top_n = min(10, len(df_sorted))
    top_runs = df_sorted.head(top_n)
    
    plt.figure(figsize=(14, 8))
    x_pos = np.arange(top_n)
    plt.barh(x_pos, top_runs['best_val_acc'].values, alpha=0.7)
    plt.yticks(x_pos, [f"LR={r['lr']:.4f}, WD={r['weight_decay']:.0e}\nDil={r['dilation_stage3']}, Blk={r['blocks_per_stage']}" 
                       for _, r in top_runs.iterrows()], fontsize=9)
    plt.xlabel('Val Accuracy', fontsize=12)
    plt.title(f'Top {top_n} Combinaisons - Val Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    plot_path2 = os.path.join(artifacts_dir, 'grid_search_top_combinations.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique top combinaisons: {plot_path2}")
    
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"✓ Grid search terminée")
    print(f"  - {total_combinations} combinaisons testées")
    print(f"  - {args.epochs} époques par run")
    print(f"  - Meilleure Val Accuracy: {best_run['best_val_acc']:.4f}")
    print(f"\n✓ Résultats sauvegardés:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Graphiques: {artifacts_dir}/grid_search_*.png")
    print(f"  - Logs TensorBoard: {runs_dir}/grid_*")


if __name__ == "__main__":
    main()
