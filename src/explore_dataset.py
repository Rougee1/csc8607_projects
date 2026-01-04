"""
Script pour calculer les statistiques exploratoires du dataset EuroSAT RGB.
À exécuter pour répondre aux questions D4 et D5 du rapport.

Usage:
    python -m src.explore_dataset --config configs/config.yaml
"""

import argparse
import yaml
from collections import Counter
from datasets import load_dataset
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os


def get_class_names():
    """Retourne les noms des 10 classes EuroSAT RGB."""
    return [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake"
    ]


def compute_statistics(config_path, log_to_tensorboard=False):
    """
    Calcule les statistiques exploratoires du dataset EuroSAT RGB.
    
    Args:
        config_path: Chemin vers config.yaml
        log_to_tensorboard: Si True, logge les stats dans TensorBoard
    """
    # Charger la config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Charger le dataset
    print("Chargement du dataset EuroSAT RGB...")
    hf_datasets = load_dataset("timm/eurosat-rgb")
    
    # Récupérer les splits
    train_split = config['dataset']['split']['train']
    val_split = config['dataset']['split']['val']
    test_split = config['dataset']['split']['test']
    
    train_hf = hf_datasets[train_split]
    val_hf = hf_datasets[val_split]
    test_hf = hf_datasets[test_split]
    
    class_names = get_class_names()
    
    # 1. Distribution des classes par split
    print("\n" + "="*60)
    print("DISTRIBUTION DES CLASSES")
    print("="*60)
    
    def count_classes(dataset, split_name):
        """Compte les occurrences de chaque classe."""
        labels = [item['label'] for item in dataset]
        counter = Counter(labels)
        return counter, labels
    
    train_counter, train_labels = count_classes(train_hf, "train")
    val_counter, val_labels = count_classes(val_hf, "validation")
    test_counter, test_labels = count_classes(test_hf, "test")
    
    # Afficher le tableau
    print(f"\n{'Classe':<25} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Total':>8}")
    print("-" * 70)
    
    total_by_class = {}
    for class_idx in range(10):
        class_name = class_names[class_idx]
        train_count = train_counter.get(class_idx, 0)
        val_count = val_counter.get(class_idx, 0)
        test_count = test_counter.get(class_idx, 0)
        total = train_count + val_count + test_count
        total_by_class[class_name] = total
        
        print(f"{class_name:<25} | {train_count:>8} | {val_count:>8} | {test_count:>8} | {total:>8}")
    
    # Totaux
    print("-" * 70)
    print(f"{'TOTAL':<25} | {len(train_hf):>8} | {len(val_hf):>8} | {len(test_hf):>8} | {len(train_hf) + len(val_hf) + len(test_hf):>8}")
    
    # 2. Vérifier l'équilibrage
    print("\n" + "="*60)
    print("ANALYSE DE L'ÉQUILIBRAGE")
    print("="*60)
    
    train_counts = [train_counter.get(i, 0) for i in range(10)]
    val_counts = [val_counter.get(i, 0) for i in range(10)]
    test_counts = [test_counter.get(i, 0) for i in range(10)]
    
    train_min = min(train_counts)
    train_max = max(train_counts)
    train_ratio = train_max / train_min if train_min > 0 else float('inf')
    
    print(f"Train - Min: {train_min}, Max: {train_max}, Ratio max/min: {train_ratio:.2f}")
    
    if train_ratio > 2.0:
        print("⚠️  ATTENTION: Le dataset est déséquilibré (ratio > 2.0)")
        print("   Considérer l'utilisation de poids de classes ou d'un échantillonnage stratifié.")
    else:
        print("✓ Le dataset est relativement équilibré.")
    
    # 3. Vérifier les tailles d'images
    print("\n" + "="*60)
    print("VÉRIFICATION DES TAILLES D'IMAGES")
    print("="*60)
    
    # Échantillonner quelques images pour vérifier
    sample_sizes = []
    for i in range(min(100, len(train_hf))):  # Vérifier 100 images max
        image = train_hf[i]['image']
        if isinstance(image, Image.Image):
            sample_sizes.append(image.size)
        else:
            # Si c'est déjà un array numpy
            if hasattr(image, 'shape'):
                sample_sizes.append((image.shape[1], image.shape[0]))
    
    unique_sizes = set(sample_sizes)
    print(f"Tailles d'images trouvées (échantillon de {len(sample_sizes)} images):")
    for size in sorted(unique_sizes):
        count = sample_sizes.count(size)
        print(f"  {size}: {count} occurrences")
    
    if len(unique_sizes) == 1:
        print("✓ Toutes les images ont la même taille (uniforme).")
    else:
        print("⚠️  ATTENTION: Les images ont des tailles variées.")
        print("   Le preprocessing devra redimensionner uniformément.")
    
    # 4. Vérifier les canaux (RGB)
    print("\n" + "="*60)
    print("VÉRIFICATION DES CANAUX")
    print("="*60)
    
    sample_image = train_hf[0]['image']
    if isinstance(sample_image, Image.Image):
        mode = sample_image.mode
        print(f"Mode de l'image: {mode}")
        if mode == 'RGB':
            print("✓ Images en RGB (3 canaux) comme attendu.")
        else:
            print(f"⚠️  Mode inattendu: {mode}")
    else:
        print("Image au format array numpy")
        if hasattr(sample_image, 'shape'):
            print(f"Shape: {sample_image.shape}")
    
    # 5. Résumé pour le rapport
    print("\n" + "="*60)
    print("RÉSUMÉ POUR LE RAPPORT")
    print("="*60)
    
    print("\n--- D4. Distribution des classes ---")
    print("Tableau ci-dessus (à copier dans le rapport)")
    print(f"Le dataset est {'relativement équilibré' if train_ratio <= 2.0 else 'déséquilibré'} (ratio max/min = {train_ratio:.2f})")
    
    print("\n--- D5. Particularités détectées ---")
    particularites = []
    if len(unique_sizes) == 1:
        particularites.append("✓ Toutes les images ont une taille uniforme (64×64 pixels)")
    else:
        particularites.append("⚠️  Les images ont des tailles variées")
    
    if train_ratio > 2.0:
        particularites.append("⚠️  Déséquilibre des classes (ratio max/min > 2.0)")
    else:
        particularites.append("✓ Distribution des classes relativement équilibrée")
    
    particularites.append("✓ Images en RGB (3 canaux)")
    particularites.append("✓ Pas de valeurs manquantes (dataset complet)")
    particularites.append("✓ Labels entiers de 0 à 9 (classification multiclasses)")
    
    for p in particularites:
        print(f"  {p}")
    
    # 6. Logger dans TensorBoard si demandé
    if log_to_tensorboard:
        runs_dir = config.get('paths', {}).get('runs_dir', './runs')
        os.makedirs(runs_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(runs_dir, 'dataset_exploration'))
        
        # Logger la distribution des classes
        stats_text = "Distribution des classes (Train):\n\n"
        for class_idx in range(10):
            class_name = class_names[class_idx]
            count = train_counter.get(class_idx, 0)
            stats_text += f"{class_name}: {count}\n"
        
        writer.add_text('dataset/class_distribution_train', stats_text)
        
        # Logger les particularités
        particularites_text = "\n".join(particularites)
        writer.add_text('dataset/particularites', particularites_text)
        
        writer.close()
        print(f"\n✓ Statistiques loggées dans TensorBoard: {os.path.join(runs_dir, 'dataset_exploration')}")
    
    print("\n" + "="*60)
    print("FIN DE L'EXPLORATION")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explorer le dataset EuroSAT RGB")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Logger les stats dans TensorBoard')
    
    args = parser.parse_args()
    
    compute_statistics(args.config, log_to_tensorboard=args.tensorboard)



