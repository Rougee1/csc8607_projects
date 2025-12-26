"""
Évaluation finale sur le test set.

Usage:
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from src.model import build_model
from src.utils import set_seed, get_device
from src.data_loading import get_dataloaders


def compute_accuracy(logits, labels):
    """Calcule l'accuracy."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def evaluate_model(model, test_loader, criterion, device):
    """Évalue le modèle sur le test set."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Probabilités (softmax)
            probs = torch.softmax(logits, dim=1)
            
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, labels)
            num_batches += 1
            
            # Stocker pour métriques détaillées
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc, np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Génère et sauvegarde une matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, label='Nombre d\'exemples')
    
    # Ajouter les annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Prédictions', fontsize=12, fontweight='bold')
    plt.ylabel('Vraies classes', fontsize=12, fontweight='bold')
    plt.title('Matrice de confusion - Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Matrice de confusion sauvegardée: {output_path}")


def get_class_names():
    """Retourne les noms des classes EuroSAT."""
    return [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]


def main():
    parser = argparse.ArgumentParser(description="Évaluation finale sur test set")
    parser.add_argument('--config', type=str, required=True,
                       help='Chemin vers config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint (ex: artifacts/best.ckpt)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed pour reproductibilité (défaut: depuis config)')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Logger les résultats dans TensorBoard')
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("ÉVALUATION FINALE SUR TEST SET (Section 2.9)")
    print("="*60)
    
    # Seed
    seed = args.seed if args.seed is not None else config['train']['seed']
    set_seed(seed)
    print(f"\n✓ Seed fixée à {seed}")
    
    # Device
    device = get_device(config['train'].get('device', 'auto'))
    print(f"✓ Device: {device}")
    
    # Vérifier que le checkpoint existe
    if not os.path.exists(args.checkpoint):
        print(f"❌ Erreur: Le checkpoint '{args.checkpoint}' n'existe pas.")
        return
    
    print(f"\n✓ Checkpoint trouvé: {args.checkpoint}")
    
    # Charger les données (test seulement, sans augmentation)
    print("\nChargement des données...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    num_classes = meta['num_classes']
    print(f"✓ DataLoaders chargés")
    print(f"  Nombre de classes: {num_classes}")
    print(f"  Test set: {len(test_loader.dataset)} exemples")
    
    # Construire le modèle
    print("\nConstruction du modèle...")
    model = build_model(config)
    model = model.to(device)
    print(f"✓ Modèle construit")
    
    # Charger le checkpoint
    print(f"\nChargement du checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('val_acc', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    print(f"✓ Checkpoint chargé")
    print(f"  Époque: {epoch}")
    print(f"  Val Accuracy: {val_acc:.4f}" if isinstance(val_acc, float) else f"  Val Accuracy: {val_acc}")
    print(f"  Val Loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"  Val Loss: {val_loss}")
    
    # Critère de perte
    criterion = nn.CrossEntropyLoss()
    
    # Évaluation sur le test set
    print(f"\n{'='*60}")
    print("ÉVALUATION SUR TEST SET")
    print(f"{'='*60}")
    
    test_loss, test_acc, predictions, labels, probs = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print(f"\n✓ Évaluation terminée")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Rapport de classification détaillé
    class_names = get_class_names()
    print(f"\n{'='*60}")
    print("RAPPORT DE CLASSIFICATION")
    print(f"{'='*60}")
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print(report)
    
    # Matrice de confusion
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    cm_path = os.path.join(artifacts_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(labels, predictions, class_names, cm_path)
    
    # TensorBoard (optionnel)
    if args.tensorboard:
        runs_dir = config['paths']['runs_dir']
        log_dir = os.path.join(runs_dir, f'evaluate_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        writer = SummaryWriter(log_dir)
        writer.add_scalar('test/loss', test_loss, 0)
        writer.add_scalar('test/accuracy', test_acc, 0)
        writer.close()
        print(f"\n✓ Résultats loggés dans TensorBoard: {log_dir}")
    
    # Comparaison avec validation
    print(f"\n{'='*60}")
    print("COMPARAISON TEST vs VALIDATION")
    print(f"{'='*60}")
    if isinstance(val_acc, float):
        diff = test_acc - val_acc
        diff_pct = diff * 100
        print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Différence:          {diff:+.4f} ({diff_pct:+.2f}%)")
        
        if abs(diff) < 0.01:
            print(f"\n  ✓ Écart très faible (< 1%), excellent signe de généralisation")
        elif abs(diff) < 0.02:
            print(f"\n  ✓ Écart faible (< 2%), bon signe de généralisation")
        elif diff < -0.05:
            print(f"\n  ⚠️  Écart significatif (> 5%), possible sur-apprentissage")
        else:
            print(f"\n  ℹ️  Écart modéré, à analyser selon le contexte")
    
    # Résumé pour le rapport
    print(f"\n{'='*60}")
    print("RÉSUMÉ POUR LE RAPPORT (Section M9)")
    print(f"{'='*60}")
    print(f"--- M9. Résultats test ---")
    print(f"Métrique principale (Accuracy): {test_acc:.4f} ({test_acc*100:.2f}%)")
    if isinstance(val_acc, float):
        print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Écart test - validation: {diff:+.4f} ({diff_pct:+.2f}%)")
    print(f"\nMatrice de confusion: {cm_path}")
    print(f"\nInterprétation:")
    if isinstance(val_acc, float):
        if abs(diff) < 0.01:
            print(f"  L'écart entre test ({test_acc*100:.2f}%) et validation ({val_acc*100:.2f}%)")
            print(f"  est très faible (< 1%), ce qui indique une excellente généralisation.")
            print(f"  Le modèle n'a pas sur-appris et performe de manière cohérente sur")
            print(f"  les données non vues pendant l'entraînement.")
        elif abs(diff) < 0.02:
            print(f"  L'écart entre test ({test_acc*100:.2f}%) et validation ({val_acc*100:.2f}%)")
            print(f"  est faible (< 2%), ce qui indique une bonne généralisation.")
            print(f"  Le modèle généralise bien aux données de test.")
        else:
            print(f"  L'écart entre test ({test_acc*100:.2f}%) et validation ({val_acc*100:.2f}%)")
            print(f"  est de {abs(diff_pct):.2f}%. À analyser selon le contexte.")
    
    print(f"\n{'='*60}")
    print("FIN DE L'ÉVALUATION")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
