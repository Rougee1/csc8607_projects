"""
Script pour créer les snapshots de config manquants pour les runs existants.

Ce script parcourt tous les runs de grid search et crée les config_snapshot.yaml
manquants en extrayant les hyperparamètres du nom du run.

Usage:
    python -m src.create_missing_snapshots --config configs/config.yaml
"""

import argparse
import yaml
import os
import re
from pathlib import Path
from src.utils import save_config_snapshot


def parse_run_name(run_name):
    """
    Extrait les hyperparamètres du nom d'un run.
    
    Exemple: "grid_lr=0.0005_wd=1e-05_dil=2_blk=3"
    -> {"lr": 0.0005, "weight_decay": 1e-5, "dilation_stage3": 2, "blocks_per_stage": 3}
    """
    params = {}
    
    # Pattern pour extraire les valeurs
    lr_match = re.search(r'lr=([0-9.]+)', run_name)
    wd_match = re.search(r'wd=([0-9.e-]+)', run_name)
    dil_match = re.search(r'dil=([0-9]+)', run_name)
    blk_match = re.search(r'blk=([0-9]+)', run_name)
    
    if lr_match:
        params['lr'] = float(lr_match.group(1))
    if wd_match:
        # Convertir "1e-05" en float
        wd_str = wd_match.group(1)
        if 'e' in wd_str.lower():
            params['weight_decay'] = float(wd_str)
        else:
            params['weight_decay'] = float(wd_str)
    if dil_match:
        params['dilation_stage3'] = int(dil_match.group(1))
    if blk_match:
        params['blocks_per_stage'] = int(blk_match.group(1))
    
    return params


def create_snapshot_for_run(run_dir, base_config, run_name):
    """
    Crée un snapshot de config pour un run donné.
    
    Args:
        run_dir: Chemin du répertoire du run
        base_config: Configuration de base (dict)
        run_name: Nom du run (pour extraire les hyperparamètres)
    """
    # Extraire les hyperparamètres du nom
    params = parse_run_name(run_name)
    
    if not params:
        print(f"  ⚠ Impossible d'extraire les hyperparamètres de: {run_name}")
        return False
    
    # Créer une copie de la config de base
    run_config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    
    # Mettre à jour les hyperparamètres
    if 'lr' in params:
        run_config['train']['optimizer']['lr'] = params['lr']
    if 'weight_decay' in params:
        run_config['train']['optimizer']['weight_decay'] = params['weight_decay']
    if 'dilation_stage3' in params:
        run_config['model']['dilation_stage3'] = params['dilation_stage3']
    if 'blocks_per_stage' in params:
        run_config['model']['blocks_per_stage'] = params['blocks_per_stage']
    
    # Sauvegarder le snapshot
    save_config_snapshot(run_config, run_dir)
    return True


def main():
    parser = argparse.ArgumentParser(description="Créer les snapshots de config manquants")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Chemin vers le fichier de configuration')
    args = parser.parse_args()
    
    # Charger la config de base
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    runs_dir = base_config.get('paths', {}).get('runs_dir', './runs')
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        print(f"❌ Répertoire runs introuvable: {runs_dir}")
        return
    
    print("=" * 60)
    print("CRÉATION DES SNAPSHOTS MANQUANTS")
    print("=" * 60)
    print(f"Répertoire runs: {runs_dir}")
    print()
    
    # Parcourir tous les runs de grid search
    grid_runs = list(runs_path.glob('grid_*'))
    refined_grid_runs = list(runs_path.glob('refined_grid_*'))
    all_grid_runs = grid_runs + refined_grid_runs
    
    if not all_grid_runs:
        print("⚠ Aucun run de grid search trouvé")
        return
    
    print(f"✓ {len(all_grid_runs)} runs de grid search trouvés")
    print()
    
    created = 0
    skipped = 0
    errors = 0
    
    for run_path in sorted(all_grid_runs):
        run_name = run_path.name
        snapshot_path = run_path / 'config_snapshot.yaml'
        
        # Vérifier si le snapshot existe déjà
        if snapshot_path.exists():
            print(f"  ⏭ {run_name}: snapshot déjà présent")
            skipped += 1
            continue
        
        # Créer le snapshot
        print(f"  📝 {run_name}: création du snapshot...")
        try:
            if create_snapshot_for_run(str(run_path), base_config, run_name):
                created += 1
                print(f"     ✓ Snapshot créé")
            else:
                errors += 1
                print(f"     ✗ Erreur lors de la création")
        except Exception as e:
            errors += 1
            print(f"     ✗ Erreur: {e}")
    
    print()
    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"✓ Snapshots créés: {created}")
    print(f"⏭ Snapshots déjà présents: {skipped}")
    print(f"✗ Erreurs: {errors}")
    print(f"📊 Total: {len(all_grid_runs)} runs")
    print()


if __name__ == '__main__':
    main()

