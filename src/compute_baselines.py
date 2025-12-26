"""
Calcul des baselines (section 2.0) : classe majoritaire et prédiction aléatoire.

Usage:
    python -m src.compute_baselines --config configs/config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
import random
from collections import Counter
from src.data_loading import get_dataloaders


def compute_majority_class_baseline(train_loader, val_loader, test_loader):
    """
    Calcule la performance de la baseline "classe majoritaire".
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders PyTorch
    
    Returns:
        dict avec les accuracies sur train/val/test
    """
    # Collecter tous les labels du train pour trouver la classe majoritaire
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.extend(labels.numpy().tolist())
    
    # Trouver la classe majoritaire
    label_counter = Counter(all_train_labels)
    majority_class = label_counter.most_common(1)[0][0]
    majority_count = label_counter[majority_class]
    total_count = len(all_train_labels)
    majority_ratio = majority_count / total_count
    
    print(f"\nClasse majoritaire: {majority_class} ({majority_count}/{total_count} = {majority_ratio:.2%})")
    
    # Calculer l'accuracy en prédisant toujours la classe majoritaire
    def compute_accuracy(loader, predicted_class):
        """Calcule l'accuracy en prédisant toujours predicted_class."""
        correct = 0
        total = 0
        
        for _, labels in loader:
            # Prédire toujours la classe majoritaire
            predictions = torch.full_like(labels, predicted_class)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    train_acc = compute_accuracy(train_loader, majority_class)
    val_acc = compute_accuracy(val_loader, majority_class)
    test_acc = compute_accuracy(test_loader, majority_class)
    
    return {
        'majority_class': int(majority_class),
        'majority_count': majority_count,
        'total_count': total_count,
        'majority_ratio': majority_ratio,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc
    }


def compute_random_baseline(val_loader, test_loader, num_classes=10, seed=42):
    """
    Calcule la performance de la baseline "prédiction aléatoire uniforme".
    
    Args:
        val_loader, test_loader: DataLoaders PyTorch
        num_classes: nombre de classes (10 pour EuroSAT)
        seed: seed pour reproductibilité
    
    Returns:
        dict avec les accuracies sur val/test
    """
    # Fixer la seed pour reproductibilité
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    def compute_random_accuracy(loader, num_classes):
        """Calcule l'accuracy avec prédictions aléatoires uniformes."""
        correct = 0
        total = 0
        
        for _, labels in loader:
            # Prédictions aléatoires uniformes (chaque classe a proba 1/num_classes)
            batch_size = labels.size(0)
            # Générer sur CPU (plus simple, labels sont généralement sur CPU)
            predictions = torch.randint(0, num_classes, (batch_size,))
            # S'assurer que predictions et labels sont sur le même device
            if labels.is_cuda:
                predictions = predictions.to(labels.device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    val_acc = compute_random_accuracy(val_loader, num_classes)
    test_acc = compute_random_accuracy(test_loader, num_classes)
    
    # Accuracy théorique attendue
    theoretical_accuracy = 1.0 / num_classes
    
    return {
        'theoretical_accuracy': theoretical_accuracy,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc
    }


def main():
    parser = argparse.ArgumentParser(description="Calculer les baselines")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers config.yaml')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour reproductibilité')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("CALCUL DES BASELINES (Section 2.0)")
    print("="*60)
    
    # Fixer la seed pour reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Charger les données
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    num_classes = meta['num_classes']
    
    print(f"✓ DataLoaders chargés")
    print(f"  Nombre de classes: {num_classes}")
    
    # 1. Baseline classe majoritaire
    print("\n" + "="*60)
    print("1. BASELINE : CLASSE MAJORITAIRE")
    print("="*60)
    
    majority_results = compute_majority_class_baseline(train_loader, val_loader, test_loader)
    
    print(f"\nRésultats:")
    print(f"  Train accuracy:   {majority_results['train_accuracy']:.4f} ({majority_results['train_accuracy']*100:.2f}%)")
    print(f"  Val accuracy:     {majority_results['val_accuracy']:.4f} ({majority_results['val_accuracy']*100:.2f}%)")
    print(f"  Test accuracy:     {majority_results['test_accuracy']:.4f} ({majority_results['test_accuracy']*100:.2f}%)")
    
    # 2. Baseline prédiction aléatoire
    print("\n" + "="*60)
    print("2. BASELINE : PRÉDICTION ALÉATOIRE UNIFORME")
    print("="*60)
    
    random_results = compute_random_baseline(val_loader, test_loader, num_classes, args.seed)
    
    print(f"\nRésultats:")
    print(f"  Accuracy théorique: {random_results['theoretical_accuracy']:.4f} ({random_results['theoretical_accuracy']*100:.2f}%)")
    print(f"  Val accuracy:       {random_results['val_accuracy']:.4f} ({random_results['val_accuracy']*100:.2f}%)")
    print(f"  Test accuracy:      {random_results['test_accuracy']:.4f} ({random_results['test_accuracy']*100:.2f}%)")
    
    # Résumé pour le rapport
    print("\n" + "="*60)
    print("RÉSUMÉ POUR LE RAPPORT (Section M0)")
    print("="*60)
    
    print(f"""
--- M0. Baselines ---

Classe majoritaire:
  - Classe: {majority_results['majority_class']}
  - Performance: {majority_results['val_accuracy']:.4f} ({majority_results['val_accuracy']*100:.2f}%)
  - (Prédire toujours la classe {majority_results['majority_class']})

Prédiction aléatoire uniforme:
  - Performance théorique: {random_results['theoretical_accuracy']:.4f} ({random_results['theoretical_accuracy']*100:.2f}%)
  - Performance observée (val): {random_results['val_accuracy']:.4f} ({random_results['val_accuracy']*100:.2f}%)
  - Performance observée (test): {random_results['test_accuracy']:.4f} ({random_results['test_accuracy']*100:.2f}%)

Commentaire:
  La classe majoritaire atteint {majority_results['val_accuracy']*100:.2f}% d'accuracy, ce qui représente
  la performance minimale à dépasser. La prédiction aléatoire donne environ {random_results['theoretical_accuracy']*100:.2f}%
  (1/{num_classes}), ce qui constitue un plancher théorique. Notre modèle devra dépasser ces deux baselines
  pour démontrer qu'il apprend effectivement des patterns dans les données.
    """)
    
    print("="*60)
    print("FIN DU CALCUL DES BASELINES")
    print("="*60)


if __name__ == "__main__":
    main()

