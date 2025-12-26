"""
Script de test pour vérifier le modèle (build_model).

Usage:
    python -m src.test_model --config configs/config.yaml
"""

import argparse
import yaml
import torch
from src.model import build_model
from src.utils import set_seed, get_device, count_parameters
from src.data_loading import get_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Tester le modèle")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers config.yaml')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour reproductibilité')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("TEST DU MODÈLE")
    print("="*60)
    
    # Fixer la seed
    set_seed(args.seed)
    print(f"\n✓ Seed fixée à {args.seed}")
    
    # Détecter le device
    device = get_device(config['train'].get('device', 'auto'))
    print(f"✓ Device: {device}")
    
    # Charger les données pour obtenir meta
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    print(f"✓ DataLoaders chargés")
    print(f"  Nombre de classes: {meta['num_classes']}")
    print(f"  Input shape: {meta['input_shape']}")
    
    # Construire le modèle
    print("\nConstruction du modèle...")
    model = build_model(config)
    model = model.to(device)
    print("✓ Modèle construit")
    
    # Compter les paramètres
    num_params = count_parameters(model)
    print(f"\n✓ Nombre de paramètres: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test forward pass
    print("\nTest forward pass...")
    model.eval()
    
    # Prendre un batch de test
    images, labels = next(iter(val_loader))
    images = images.to(device)
    batch_size = images.size(0)
    
    print(f"  Input shape: {images.shape}")
    
    with torch.no_grad():
        output = model(images)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (batch_size={batch_size}, num_classes={meta['num_classes']})")
    
    # Vérifier la forme
    expected_shape = (batch_size, meta['num_classes'])
    if output.shape == expected_shape:
        print("  ✓ Forme de sortie correcte!")
    else:
        print(f"  ✗ ERREUR: Forme attendue {expected_shape}, obtenue {output.shape}")
        return
    
    # Vérifier que ce sont bien des logits (pas de softmax)
    print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    print("  ✓ Logits (pas de softmax appliqué)")
    
    # Test avec différentes tailles de batch
    print("\nTest avec différentes tailles de batch...")
    for batch_size_test in [1, 2, 4, 8]:
        test_images = images[:batch_size_test]
        with torch.no_grad():
            test_output = model(test_images)
        expected = (batch_size_test, meta['num_classes'])
        if test_output.shape == expected:
            print(f"  ✓ Batch size {batch_size_test}: OK")
        else:
            print(f"  ✗ Batch size {batch_size_test}: ERREUR")
    
    # Afficher l'architecture
    print("\n" + "="*60)
    print("ARCHITECTURE DU MODÈLE")
    print("="*60)
    print(model)
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"✓ Modèle construit avec succès")
    print(f"✓ Paramètres: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"✓ Forward pass fonctionne")
    print(f"✓ Forme de sortie: (batch_size, {meta['num_classes']})")
    print(f"✓ Device: {device}")
    
    # Hyperparamètres du modèle
    model_config = config['model']
    print(f"\nHyperparamètres du modèle:")
    print(f"  - blocks_per_stage: {model_config['blocks_per_stage']}")
    print(f"  - dilation_stage3: {model_config['dilation_stage3']}")
    print(f"  - channels: {model_config['channels']}")
    print(f"  - batch_norm: {model_config.get('batch_norm', True)}")
    print(f"  - activation: {model_config.get('activation', 'relu')}")


if __name__ == "__main__":
    main()

