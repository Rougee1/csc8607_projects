"""
Script pour comparer les courbes d'entraînement de différents runs.

Usage:
    python -m src.compare_curves --config configs/config.yaml
"""

import argparse
import yaml
import os
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalars_from_tensorboard(log_dir, tags):
    """
    Extrait les scalaires depuis un répertoire TensorBoard.
    
    Args:
        log_dir: Chemin vers le répertoire de logs TensorBoard
        tags: Liste des tags à extraire (ex: ['train/loss', 'val/loss'])
    
    Returns:
        dict: {tag: {'steps': [...], 'values': [...]}}
    """
    if not os.path.exists(log_dir):
        return None
    
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        scalars = {}
        for tag in tags:
            if tag in ea.Tags()['scalars']:
                scalar_events = ea.Scalars(tag)
                steps = [e.step for e in scalar_events]
                values = [e.value for e in scalar_events]
                scalars[tag] = {'steps': steps, 'values': values}
            else:
                scalars[tag] = {'steps': [], 'values': []}
        
        return scalars
    except Exception as e:
        print(f"⚠️  Erreur lors de la lecture de {log_dir}: {e}")
        return None


def parse_run_name(run_name):
    """
    Parse le nom d'un run pour extraire les hyperparamètres.
    
    Format attendu: grid_lr=0.0005_wd=1e-05_dil=2_blk=3
    """
    match = re.match(r'grid_lr=([\d.]+)_wd=([\d.e-]+)_dil=(\d+)_blk=(\d+)', run_name)
    if match:
        lr = float(match.group(1))
        wd_str = match.group(2)
        # Convertir '1e-05' en float
        wd = float(wd_str) if 'e' not in wd_str.lower() else float(wd_str)
        dilation = int(match.group(3))
        blocks = int(match.group(4))
        return {'lr': lr, 'weight_decay': wd, 'dilation_stage3': dilation, 'blocks_per_stage': blocks}
    return None


def find_grid_search_runs(runs_dir):
    """Trouve tous les runs de grid search."""
    runs = []
    if not os.path.exists(runs_dir):
        return runs
    
    for run_dir in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_dir)
        if os.path.isdir(run_path) and run_dir.startswith('grid_'):
            hparams = parse_run_name(run_dir)
            if hparams:
                runs.append({
                    'name': run_dir,
                    'path': run_path,
                    'hparams': hparams
                })
    
    return runs


def plot_lr_comparison(runs_data, output_path):
    """
    Compare l'effet du learning rate sur train/loss au début d'entraînement.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Grouper par LR (même weight_decay, dilation, blocks)
    # On prend la meilleure combinaison (WD=1e-5, D=2, Blocks=3) et on varie LR
    base_wd = 1e-5
    base_dilation = 2
    base_blocks = 3
    
    lr_values = sorted(set([r['hparams']['lr'] for r in runs_data]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(lr_values)))
    
    for lr, color in zip(lr_values, colors):
        # Trouver les runs avec ce LR et les hyperparamètres de base
        matching_runs = [
            r for r in runs_data
            if (r['hparams']['lr'] == lr and
                abs(r['hparams']['weight_decay'] - base_wd) < 1e-8 and
                r['hparams']['dilation_stage3'] == base_dilation and
                r['hparams']['blocks_per_stage'] == base_blocks)
        ]
        
        if matching_runs:
            run = matching_runs[0]
            scalars = extract_scalars_from_tensorboard(run['path'], ['train/loss'])
            if scalars and 'train/loss' in scalars and scalars['train/loss']['steps']:
                steps = scalars['train/loss']['steps']
                values = scalars['train/loss']['values']
                # Limiter aux 5 premières époques pour voir l'effet au début
                if len(steps) > 5:
                    steps = steps[:5]
                    values = values[:5]
                ax.plot(steps, values, label=f'LR={lr:.4f}', color=color, linewidth=2, marker='o')
    
    ax.set_xlabel('Époque', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('Comparaison des Learning Rates - Train Loss (5 premières époques)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique LR sauvegardé: {output_path}")


def plot_weight_decay_comparison(runs_data, output_path):
    """
    Compare l'effet du weight decay sur l'écart train/val.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Grouper par weight decay (même LR, dilation, blocks)
    # On prend la meilleure combinaison (LR=0.0005, D=2, Blocks=3) et on varie WD
    base_lr = 0.0005
    base_dilation = 2
    base_blocks = 3
    
    wd_values = sorted(set([r['hparams']['weight_decay'] for r in runs_data]))
    colors = plt.cm.plasma(np.linspace(0, 1, len(wd_values)))
    
    for wd, color in zip(wd_values, colors):
        # Trouver les runs avec ce WD et les hyperparamètres de base
        matching_runs = [
            r for r in runs_data
            if (abs(r['hparams']['lr'] - base_lr) < 1e-6 and
                abs(r['hparams']['weight_decay'] - wd) < 1e-8 and
                r['hparams']['dilation_stage3'] == base_dilation and
                r['hparams']['blocks_per_stage'] == base_blocks)
        ]
        
        if matching_runs:
            run = matching_runs[0]
            scalars = extract_scalars_from_tensorboard(run['path'], ['train/loss', 'val/loss'])
            if scalars and 'train/loss' in scalars and 'val/loss' in scalars:
                steps = scalars['train/loss']['steps']
                train_loss = scalars['train/loss']['values']
                val_loss = scalars['val/loss']['values']
                
                if steps:
                    # Graphique 1: Loss train vs val
                    ax1.plot(steps, train_loss, label=f'Train (WD={wd:.0e})', 
                            color=color, linewidth=2, linestyle='-', marker='o', markersize=4)
                    ax1.plot(steps, val_loss, label=f'Val (WD={wd:.0e})', 
                            color=color, linewidth=2, linestyle='--', marker='s', markersize=4)
                    
                    # Graphique 2: Écart train/val
                    gap = [v - t for t, v in zip(train_loss, val_loss)]
                    ax2.plot(steps, gap, label=f'WD={wd:.0e}', color=color, linewidth=2, marker='o')
    
    ax1.set_xlabel('Époque', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Train vs Val Loss selon Weight Decay', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Époque', fontsize=12)
    ax2.set_ylabel('Écart Val - Train Loss', fontsize=12)
    ax2.set_title('Écart Train/Val selon Weight Decay', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique Weight Decay sauvegardé: {output_path}")


def plot_model_hparams_comparison(runs_data, output_path):
    """
    Compare l'effet des hyperparamètres du modèle (dilation_stage3 et blocks_per_stage).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Grouper par hyperparamètres du modèle (même LR, WD)
    base_lr = 0.0005
    base_wd = 1e-5
    
    # Graphique 1: Comparaison dilation_stage3 (même blocks_per_stage=3)
    dilation_values = sorted(set([r['hparams']['dilation_stage3'] for r in runs_data]))
    colors_dil = plt.cm.coolwarm(np.linspace(0, 1, len(dilation_values)))
    
    for dilation, color in zip(dilation_values, colors_dil):
        matching_runs = [
            r for r in runs_data
            if (abs(r['hparams']['lr'] - base_lr) < 1e-6 and
                abs(r['hparams']['weight_decay'] - base_wd) < 1e-8 and
                r['hparams']['dilation_stage3'] == dilation and
                r['hparams']['blocks_per_stage'] == 3)
        ]
        
        if matching_runs:
            run = matching_runs[0]
            scalars = extract_scalars_from_tensorboard(run['path'], ['train/loss', 'val/accuracy'])
            if scalars:
                steps = scalars.get('train/loss', {}).get('steps', [])
                if steps:
                    train_loss = scalars.get('train/loss', {}).get('values', [])
                    val_acc = scalars.get('val/accuracy', {}).get('values', [])
                    
                    ax1.plot(steps, train_loss, label=f'Dilation={dilation}', 
                            color=color, linewidth=2, marker='o')
                    ax3.plot(steps, val_acc, label=f'Dilation={dilation}', 
                            color=color, linewidth=2, marker='s')
    
    # Graphique 2: Comparaison blocks_per_stage (même dilation_stage3=2)
    blocks_values = sorted(set([r['hparams']['blocks_per_stage'] for r in runs_data]))
    colors_blk = plt.cm.Set2(np.linspace(0, 1, len(blocks_values)))
    
    for blocks, color in zip(blocks_values, colors_blk):
        matching_runs = [
            r for r in runs_data
            if (abs(r['hparams']['lr'] - base_lr) < 1e-6 and
                abs(r['hparams']['weight_decay'] - base_wd) < 1e-8 and
                r['hparams']['dilation_stage3'] == 2 and
                r['hparams']['blocks_per_stage'] == blocks)
        ]
        
        if matching_runs:
            run = matching_runs[0]
            scalars = extract_scalars_from_tensorboard(run['path'], ['train/loss', 'val/accuracy'])
            if scalars:
                steps = scalars.get('train/loss', {}).get('steps', [])
                if steps:
                    train_loss = scalars.get('train/loss', {}).get('values', [])
                    val_acc = scalars.get('val/accuracy', {}).get('values', [])
                    
                    ax2.plot(steps, train_loss, label=f'Blocks={blocks}', 
                            color=color, linewidth=2, marker='o')
                    ax4.plot(steps, val_acc, label=f'Blocks={blocks}', 
                            color=color, linewidth=2, marker='s')
    
    # Configuration des graphiques
    ax1.set_xlabel('Époque', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Effet de Dilation (Blocks=3)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Époque', fontsize=12)
    ax2.set_ylabel('Train Loss', fontsize=12)
    ax2.set_title('Effet de Blocks per Stage (Dilation=2)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Époque', fontsize=12)
    ax3.set_ylabel('Val Accuracy', fontsize=12)
    ax3.set_title('Effet de Dilation (Blocks=3)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Époque', fontsize=12)
    ax4.set_ylabel('Val Accuracy', fontsize=12)
    ax4.set_title('Effet de Blocks per Stage (Dilation=2)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique Hyperparamètres Modèle sauvegardé: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare les courbes d\'entraînement de différents runs')
    parser.add_argument('--config', type=str, required=True, help='Chemin vers le fichier de configuration')
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    runs_dir = config['paths']['runs_dir']
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"COMPARAISON DES COURBES (Section 2.7)")
    print(f"{'='*60}")
    
    # Trouver tous les runs de grid search
    print("\nRecherche des runs de grid search...")
    runs_data = find_grid_search_runs(runs_dir)
    
    if not runs_data:
        print(f"⚠️  Aucun run de grid search trouvé dans {runs_dir}")
        print("   Assurez-vous d'avoir exécuté grid_search.py au préalable.")
        return
    
    print(f"✓ {len(runs_data)} runs trouvés")
    
    # Générer les graphiques
    print("\nGénération des graphiques comparatifs...")
    
    # 1. Comparaison LR
    plot_lr_comparison(runs_data, os.path.join(artifacts_dir, 'comparison_lr.png'))
    
    # 2. Comparaison Weight Decay
    plot_weight_decay_comparison(runs_data, os.path.join(artifacts_dir, 'comparison_weight_decay.png'))
    
    # 3. Comparaison Hyperparamètres Modèle
    plot_model_hparams_comparison(runs_data, os.path.join(artifacts_dir, 'comparison_model_hparams.png'))
    
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"✓ 3 graphiques comparatifs générés dans {artifacts_dir}/")
    print(f"  - comparison_lr.png (effet du Learning Rate)")
    print(f"  - comparison_weight_decay.png (effet du Weight Decay)")
    print(f"  - comparison_model_hparams.png (effet des hyperparamètres du modèle)")


if __name__ == '__main__':
    main()



