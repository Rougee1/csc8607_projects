"""
Vérification de la perte initiale et du premier batch (section 2.3).

Usage:
    python -m src.check_initial_loss --config configs/config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from src.model import build_model
from src.utils import set_seed, get_device
from src.data_loading import get_dataloaders


def compute_gradient_norm(model):
    """
    Calcule la norme des gradients du modèle.
    
    Returns:
        float: somme des normes L2 de tous les gradients
    """
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    return total_norm, param_count


def main():
    parser = argparse.ArgumentParser(description="Vérifier la perte initiale")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers config.yaml')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed pour reproductibilité')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("VÉRIFICATION DE LA PERTE INITIALE (Section 2.3)")
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
    input_shape = meta['input_shape']
    print(f"✓ DataLoaders chargés")
    print(f"  Nombre de classes: {num_classes}")
    print(f"  Input shape: {input_shape}")
    
    # Construire le modèle
    print("\nConstruction du modèle...")
    model = build_model(config)
    model = model.to(device)
    model.train()  # Mode entraînement (pour BatchNorm)
    print("✓ Modèle construit et en mode entraînement")
    
    # Charger un batch
    print("\nChargement d'un batch d'entraînement...")
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    batch_size = images.size(0)
    
    print(f"✓ Batch chargé")
    print(f"  Batch size: {batch_size}")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels range: [{labels.min().item()}, {labels.max().item()}]")
    print(f"  Labels (exemple): {labels[:10].cpu().numpy()}")
    
    # Forward pass
    print("\nForward pass...")
    logits = model(images)
    print(f"✓ Forward pass effectué")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  Logits mean: {logits.mean().item():.4f}")
    print(f"  Logits std: {logits.std().item():.4f}")
    
    # Calculer la perte
    print("\nCalcul de la perte...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    loss_value = loss.item()
    
    print(f"✓ Perte calculée")
    print(f"  Loss: {loss_value:.6f}")
    
    # Perte théorique attendue
    theoretical_loss = -np.log(1.0 / num_classes)
    print(f"\n  Perte théorique attendue (si logits ~0):")
    print(f"    -log(1/{num_classes}) = -log(1/{num_classes}) = {theoretical_loss:.6f}")
    
    # Vérifier la cohérence
    print(f"\n  Analyse:")
    if abs(loss_value - theoretical_loss) < 0.5:
        print(f"    ✓ Perte cohérente (différence: {abs(loss_value - theoretical_loss):.4f})")
    else:
        print(f"    ⚠️  Perte différente de la valeur théorique (différence: {abs(loss_value - theoretical_loss):.4f})")
        print(f"       Cela peut être normal si les poids ne sont pas initialisés à 0")
    
    # Calculer les probabilités (softmax) pour vérifier
    probs = torch.softmax(logits, dim=1)
    print(f"\n  Probabilités (softmax) sur le premier exemple:")
    print(f"    {probs[0].cpu().detach().numpy()}")
    print(f"    Somme: {probs[0].sum().item():.6f} (devrait être 1.0)")
    print(f"    Probabilité moyenne par classe: {probs[0].mean().item():.6f} (théorique: {1.0/num_classes:.6f})")
    
    # Backward pass
    print("\nBackward pass (rétropropagation)...")
    model.zero_grad()  # Réinitialiser les gradients
    loss.backward()
    print("✓ Backward pass effectué")
    
    # Vérifier les gradients
    print("\nVérification des gradients...")
    total_grad_norm, param_count = compute_gradient_norm(model)
    
    print(f"✓ Gradients calculés")
    print(f"  Nombre de paramètres avec gradients: {param_count}")
    print(f"  Norme totale des gradients: {total_grad_norm:.6f}")
    
    if total_grad_norm > 1e-6:
        print(f"  ✓ Gradients non-nuls (norme > 1e-6)")
    else:
        print(f"  ⚠️  Gradients très petits (norme < 1e-6) - possible problème")
    
    # Afficher quelques exemples de gradients
    print(f"\n  Exemples de gradients (normes par couche):")
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and grad_count < 5:
            grad_norm = param.grad.data.norm(2).item()
            print(f"    {name[:50]:<50} : {grad_norm:.6f}")
            grad_count += 1
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ POUR LE RAPPORT (Section M2)")
    print("="*60)
    
    # Préparer les messages conditionnels
    coherence_msg = "cohérente" if abs(loss_value - theoretical_loss) < 0.5 else "différente"
    coherence_check = "✓ OUI" if abs(loss_value - theoretical_loss) < 0.5 else "⚠️ À vérifier"
    init_msg = ("Cela indique que les poids sont initialisés de manière appropriée." 
                if abs(loss_value - theoretical_loss) < 0.5 
                else "Cela peut être dû à l'initialisation des poids (non-nulle).")
    grad_status = "bien calculés" if total_grad_norm > 1e-6 else "très petits"
    grad_check = "✓ OUI" if total_grad_norm > 1e-6 else "⚠️ NON"
    confirm_msg = "confirme" if total_grad_norm > 1e-6 else "peut indiquer un problème avec"
    
    print(f"""
--- M2. Perte initiale ---

Formes:
  - Batch images: {images.shape}
  - Batch labels: {labels.shape}
  - Sortie modèle (logits): {logits.shape}

Perte initiale:
  - Observée: {loss_value:.6f}
  - Théorique (si logits ~0): {theoretical_loss:.6f}
  - Différence: {abs(loss_value - theoretical_loss):.6f}
  - Cohérence: {coherence_check}

Gradients:
  - Norme totale: {total_grad_norm:.6f}
  - Gradients non-nuls: {grad_check}

Commentaire:
  La perte initiale de {loss_value:.6f} est {coherence_msg} 
  avec la valeur théorique de {theoretical_loss:.6f} pour {num_classes} classes. 
  {init_msg}
  Les gradients sont {grad_status}, 
  ce qui {confirm_msg} 
  le bon fonctionnement de la rétropropagation.
    """)
    
    print("="*60)
    print("FIN DE LA VÉRIFICATION")
    print("="*60)


if __name__ == "__main__":
    main()

