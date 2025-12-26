"""
Recherche de taux d'apprentissage (LR finder).

Usage:
    python -m src.lr_finder --config configs/config.yaml
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


def main():
    parser = argparse.ArgumentParser(description="LR Finder")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers config.yaml')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                       help='LR minimum (défaut: 1e-5)')
    parser.add_argument('--max_lr', type=float, default=1e-1,
                       help='LR maximum (défaut: 1e-1)')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Nombre d\'itérations pour chaque LR (défaut: 100)')
    parser.add_argument('--num_lrs', type=int, default=50,
                       help='Nombre de LR à tester (défaut: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour reproductibilité')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("LR FINDER (Section 2.4)")
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
    
    # Construire le modèle
    print("\nConstruction du modèle...")
    model = build_model(config)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"✓ Modèle construit")
    print(f"  Paramètres: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Hyperparamètres du modèle
    model_config = config['model']
    print(f"\nHyperparamètres du modèle:")
    print(f"  - blocks_per_stage: {model_config['blocks_per_stage']}")
    print(f"  - dilation_stage3: {model_config['dilation_stage3']}")
    
    # Créer un itérateur infini pour les données
    train_iter = iter(train_loader)
    
    # Générer les LR à tester (échelle logarithmique)
    lrs = np.logspace(np.log10(args.min_lr), np.log10(args.max_lr), args.num_lrs)
    
    print(f"\nParamètres du LR finder:")
    print(f"  - LR min: {args.min_lr}")
    print(f"  - LR max: {args.max_lr}")
    print(f"  - Nombre de LR: {args.num_lrs}")
    print(f"  - Itérations par LR: {args.num_steps}")
    
    # TensorBoard
    runs_dir = config['paths']['runs_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lr_finder_{timestamp}"
    log_dir = os.path.join(runs_dir, run_name)
    writer = SummaryWriter(log_dir)
    
    print(f"\nTensorBoard:")
    print(f"  - Log dir: {log_dir}")
    print(f"  - Tags: lr_finder/lr, lr_finder/loss")
    
    # Sauvegarder la config
    save_config_snapshot(config, log_dir)
    
    # Stocker les résultats
    results = {
        'lrs': [],
        'losses': [],
        'initial_losses': [],
        'final_losses': [],
        'loss_changes': []
    }
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"BALAYAGE LR ({args.num_lrs} valeurs)")
    print(f"{'='*60}")
    
    # Sauvegarder l'état initial du modèle
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    for lr_idx, lr in enumerate(lrs):
        # Réinitialiser le modèle à l'état initial
        model.load_state_dict(initial_state)
        
        # Créer un nouvel optimiseur avec ce LR
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        
        model.train()
        
        # Stocker la loss initiale
        initial_loss = None
        final_loss = None
        losses_for_lr = []
        
        # Entraîner quelques itérations avec ce LR
        for step in range(args.num_steps):
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            loss_value = loss.item()
            losses_for_lr.append(loss_value)
            
            # Logger dans TensorBoard
            global_step = lr_idx * args.num_steps + step
            writer.add_scalar('lr_finder/lr', lr, global_step)
            writer.add_scalar('lr_finder/loss', loss_value, global_step)
            
            # Stocker la première et dernière loss
            if step == 0:
                initial_loss = loss_value
            if step == args.num_steps - 1:
                final_loss = loss_value
        
        # Calculer le changement de loss
        loss_change = final_loss - initial_loss if initial_loss is not None else 0
        
        # Stocker les résultats
        results['lrs'].append(lr)
        results['losses'].append(np.mean(losses_for_lr))
        results['initial_losses'].append(initial_loss)
        results['final_losses'].append(final_loss)
        results['loss_changes'].append(loss_change)
        
        # Afficher le progrès
        if (lr_idx + 1) % 10 == 0 or lr_idx == 0 or lr_idx == len(lrs) - 1:
            print(f"LR {lr_idx+1:3d}/{args.num_lrs}: LR={lr:.2e}, Loss={np.mean(losses_for_lr):.4f}, Change={loss_change:+.4f}")
    
    writer.close()
    
    # Analyser les résultats pour trouver la zone stable
    losses_array = np.array(results['losses'])
    lrs_array = np.array(results['lrs'])
    
    # Trouver la zone où la loss diminue (change < 0) et n'explose pas (loss < 10)
    stable_mask = (np.array(results['loss_changes']) < 0) & (losses_array < 10.0)
    stable_lrs = lrs_array[stable_mask]
    stable_losses = losses_array[stable_mask]
    
    if len(stable_lrs) > 0:
        # Choisir un LR dans la zone stable (par exemple, celui avec la meilleure diminution)
        best_idx = np.argmin(np.array(results['loss_changes'])[stable_mask])
        recommended_lr = stable_lrs[best_idx]
        recommended_loss = stable_losses[best_idx]
    else:
        # Si aucune zone stable trouvée, prendre le LR avec la meilleure loss
        best_idx = np.argmin(losses_array)
        recommended_lr = lrs_array[best_idx]
        recommended_loss = losses_array[best_idx]
    
    print(f"\n{'='*60}")
    print("ANALYSE DES RÉSULTATS")
    print(f"{'='*60}")
    print(f"✓ LR recommandé: {recommended_lr:.6f} ({recommended_lr:.2e})")
    print(f"  Loss correspondante: {recommended_loss:.4f}")
    if len(stable_lrs) > 0:
        print(f"  Zone stable: LR entre {stable_lrs.min():.2e} et {stable_lrs.max():.2e}")
    else:
        print(f"  ⚠️  Aucune zone stable claire identifiée")
    
    # Générer les graphiques
    print(f"\nGénération des graphiques...")
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Graphique 1: LR vs Loss (principal)
    plt.figure(figsize=(12, 8))
    plt.semilogx(results['lrs'], results['losses'], 'b-', linewidth=2, label='Loss moyenne')
    if len(stable_lrs) > 0:
        plt.semilogx(stable_lrs, stable_losses, 'g-', linewidth=3, alpha=0.7, label='Zone stable')
    plt.axvline(recommended_lr, color='r', linestyle='--', linewidth=2, label=f'LR recommandé: {recommended_lr:.2e}')
    plt.xlabel('Learning Rate (échelle logarithmique)', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('LR Finder - Loss en fonction du Learning Rate\n(Zone stable: loss diminue sans divergence)', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plot_path1 = os.path.join(artifacts_dir, 'lr_finder_main.png')
    plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique principal: {plot_path1}")
    
    # Graphique 2: LR vs Loss avec zones identifiées
    plt.figure(figsize=(14, 8))
    
    # Identifier les zones
    too_small_mask = np.array(results['loss_changes']) > -0.1  # Loss ne diminue pas assez
    too_large_mask = losses_array > 10.0  # Loss explose
    stable_mask_plot = ~too_small_mask & ~too_large_mask
    
    plt.semilogx(lrs_array[too_small_mask], losses_array[too_small_mask], 
                'orange', linewidth=2, alpha=0.7, label='LR trop petit (loss ne diminue pas)')
    plt.semilogx(lrs_array[stable_mask_plot], losses_array[stable_mask_plot], 
                'green', linewidth=3, label='Zone stable (loss diminue)')
    plt.semilogx(lrs_array[too_large_mask], losses_array[too_large_mask], 
                'red', linewidth=2, alpha=0.7, label='LR trop grand (loss explose)')
    plt.axvline(recommended_lr, color='blue', linestyle='--', linewidth=2, 
               label=f'LR recommandé: {recommended_lr:.2e}')
    
    plt.xlabel('Learning Rate (échelle logarithmique)', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('LR Finder - Zones identifiées\n(Trop petit | Stable | Trop grand)', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    plot_path2 = os.path.join(artifacts_dir, 'lr_finder_zones.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique zones: {plot_path2}")
    
    # Graphique 3: Changement de loss (initial vs final)
    plt.figure(figsize=(12, 8))
    plt.semilogx(results['lrs'], results['initial_losses'], 'b-', linewidth=2, alpha=0.7, label='Loss initiale')
    plt.semilogx(results['lrs'], results['final_losses'], 'r-', linewidth=2, alpha=0.7, label='Loss finale')
    plt.axvline(recommended_lr, color='g', linestyle='--', linewidth=2, label=f'LR recommandé: {recommended_lr:.2e}')
    plt.xlabel('Learning Rate (échelle logarithmique)', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('LR Finder - Loss initiale vs finale\n(Meilleur LR: grande différence négative)', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plot_path3 = os.path.join(artifacts_dir, 'lr_finder_initial_final.png')
    plt.savefig(plot_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique initial/final: {plot_path3}")
    
    # Graphique 4: Changement de loss (delta)
    plt.figure(figsize=(12, 8))
    colors = ['red' if x > 0 else 'green' if x < -0.5 else 'orange' for x in results['loss_changes']]
    plt.semilogx(results['lrs'], results['loss_changes'], 'o-', linewidth=2, markersize=4, color='blue', alpha=0.7)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axvline(recommended_lr, color='g', linestyle='--', linewidth=2, label=f'LR recommandé: {recommended_lr:.2e}')
    plt.xlabel('Learning Rate (échelle logarithmique)', fontsize=14)
    plt.ylabel('Changement de Loss (finale - initiale)', fontsize=14)
    plt.title('LR Finder - Changement de Loss\n(Négatif = loss diminue, Positif = loss augmente)', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plot_path4 = os.path.join(artifacts_dir, 'lr_finder_delta.png')
    plt.savefig(plot_path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique delta: {plot_path4}")
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"✓ LR Finder terminé")
    print(f"  - LR recommandé: {recommended_lr:.6f} ({recommended_lr:.2e})")
    print(f"  - Loss correspondante: {recommended_loss:.4f}")
    print(f"  - Weight decay recommandé: 1e-4 ou 1e-5")
    print(f"\n✓ Graphiques sauvegardés dans: {artifacts_dir}")
    print(f"  - lr_finder_main.png (courbe principale)")
    print(f"  - lr_finder_zones.png (zones identifiées)")
    print(f"  - lr_finder_initial_final.png (loss initiale vs finale)")
    print(f"  - lr_finder_delta.png (changement de loss)")
    print(f"\n✓ Logs TensorBoard sauvegardés dans: {log_dir}")
    print(f"  Visualiser avec: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
