"""
Script de vérification du pipeline de données (section 1.6).

Vérifie :
- Formes des batchs
- Shuffle des DataLoaders
- Exemples visuels après preprocessing/augmentation
- Cohérence des labels

Usage:
    python -m src.verify_pipeline --config configs/config.yaml
"""

import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.data_loading import get_dataloaders


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Dénormalise une image tensor pour la visualisation.
    Inverse la normalisation ImageNet.
    """
    # Copier pour ne pas modifier l'original
    img = tensor.clone()
    
    # Dénormaliser : (img * std) + mean
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    # Clamper entre 0 et 1
    img = torch.clamp(img, 0, 1)
    
    # Convertir en numpy pour matplotlib
    img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    
    return img


def verify_dataloaders(config_path, save_images=True):
    """
    Vérifie le pipeline de données complet.
    
    Args:
        config_path: Chemin vers config.yaml
        save_images: Si True, sauvegarde des exemples d'images
    """
    # Charger la config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("VÉRIFICATION DU PIPELINE DE DONNÉES")
    print("="*60)
    
    # 1. Créer les DataLoaders
    print("\n1. Création des DataLoaders...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    
    print(f"✓ DataLoaders créés")
    print(f"  Meta: {meta}")
    
    # 2. Vérifier les formes des batchs
    print("\n2. Vérification des formes des batchs...")
    
    # Train
    train_batch = next(iter(train_loader))
    train_images, train_labels = train_batch
    batch_size = config['train']['batch_size']
    
    print(f"\nTrain batch:")
    print(f"  Images shape: {train_images.shape}")
    print(f"  Labels shape: {train_labels.shape}")
    print(f"  Expected input_shape: {meta['input_shape']}")
    
    # Vérifier que la forme correspond
    expected_shape = (batch_size, *meta['input_shape'])
    if train_images.shape == expected_shape:
        print(f"  ✓ Forme correcte: {train_images.shape} == {expected_shape}")
    else:
        print(f"  ✗ Forme incorrecte: {train_images.shape} != {expected_shape}")
    
    # Vérifier les plages de valeurs (après normalisation)
    print(f"  Image value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"  (Attendu après normalisation: environ [-2.0, 2.0])")
    
    # Val
    val_batch = next(iter(val_loader))
    val_images, val_labels = val_batch
    print(f"\nValidation batch:")
    print(f"  Images shape: {val_images.shape}")
    print(f"  Labels shape: {val_labels.shape}")
    
    # Test
    test_batch = next(iter(test_loader))
    test_images, test_labels = test_batch
    print(f"\nTest batch:")
    print(f"  Images shape: {test_images.shape}")
    print(f"  Labels shape: {test_labels.shape}")
    
    # 3. Vérifier le shuffle
    print("\n3. Vérification du shuffle...")
    
    # Prendre deux batchs consécutifs du train
    train_iter = iter(train_loader)
    batch1_images, batch1_labels = next(train_iter)
    batch2_images, batch2_labels = next(train_iter)
    
    # Vérifier si les labels sont différents (shuffle actif)
    labels1 = batch1_labels.numpy()
    labels2 = batch2_labels.numpy()
    
    if not np.array_equal(labels1, labels2):
        print(f"  ✓ Train shuffle: ACTIF (labels différents entre batchs)")
    else:
        print(f"  ✗ Train shuffle: INACTIF (mêmes labels entre batchs)")
    
    # Val ne doit pas être shuffle
    val_iter = iter(val_loader)
    val_batch1 = next(val_iter)
    val_batch2 = next(val_iter)
    # Note: on ne peut pas vraiment vérifier que val n'est pas shuffle sans relancer,
    # mais on peut vérifier que shuffle=False dans le DataLoader
    
    # 4. Vérifier les labels
    print("\n4. Vérification des labels...")
    
    # Vérifier la plage des labels
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.extend(labels.numpy().tolist())
    
    unique_labels = sorted(set(all_train_labels))
    print(f"  Labels uniques trouvés: {unique_labels}")
    print(f"  Nombre de classes attendu: {meta['num_classes']}")
    
    if len(unique_labels) == meta['num_classes'] and min(unique_labels) == 0 and max(unique_labels) == meta['num_classes'] - 1:
        print(f"  ✓ Labels cohérents: {len(unique_labels)} classes, plage [0, {max(unique_labels)}]")
    else:
        print(f"  ✗ Labels incohérents: {len(unique_labels)} classes trouvées, plage [{min(unique_labels)}, {max(unique_labels)}]")
    
    # 5. Sauvegarder des exemples visuels
    if save_images:
        print("\n5. Sauvegarde d'exemples visuels...")
        
        artifacts_dir = Path(config.get('paths', {}).get('artifacts_dir', './artifacts'))
        artifacts_dir.mkdir(exist_ok=True)
        
        # Exemples train (avec augmentation)
        print("  Exemples train (avec augmentation):")
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(min(6, len(train_images))):
            img = denormalize_image(train_images[i])
            axes[i].imshow(img)
            axes[i].set_title(f'Train - Label: {train_labels[i].item()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        train_examples_path = artifacts_dir / 'train_examples_augmented.png'
        plt.savefig(train_examples_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Sauvegardé: {train_examples_path}")
        
        # Exemples validation (sans augmentation)
        print("  Exemples validation (sans augmentation):")
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(min(6, len(val_images))):
            img = denormalize_image(val_images[i])
            axes[i].imshow(img)
            axes[i].set_title(f'Val - Label: {val_labels[i].item()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        val_examples_path = artifacts_dir / 'val_examples.png'
        plt.savefig(val_examples_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Sauvegardé: {val_examples_path}")
    
    # 6. Résumé pour le rapport
    print("\n" + "="*60)
    print("RÉSUMÉ POUR LE RAPPORT")
    print("="*60)
    
    print(f"\n--- D10. Exemples visuels ---")
    print(f"Images sauvegardées dans artifacts/:")
    print(f"  - train_examples_augmented.png (6 exemples train avec augmentation)")
    print(f"  - val_examples.png (6 exemples validation sans augmentation)")
    print(f"À insérer dans le rapport avec un commentaire.")
    
    print(f"\n--- D11. Forme des batchs ---")
    print(f"Forme exacte d'un batch train: {train_images.shape}")
    print(f"  - batch_size: {train_images.shape[0]}")
    print(f"  - canaux: {train_images.shape[1]}")
    print(f"  - hauteur: {train_images.shape[2]}")
    print(f"  - largeur: {train_images.shape[3]}")
    print(f"Meta input_shape: {meta['input_shape']}")
    print(f"  - Cohérence: {'✓ OUI' if train_images.shape[1:] == meta['input_shape'] else '✗ NON'}")
    
    print("\n" + "="*60)
    print("FIN DE LA VÉRIFICATION")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérifier le pipeline de données")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers le fichier de configuration')
    parser.add_argument('--no-images', action='store_true',
                       help='Ne pas sauvegarder les images')
    
    args = parser.parse_args()
    
    verify_dataloaders(args.config, save_images=not args.no_images)



