"""
Test complet du pipeline de données pour vérifier que tout fonctionne correctement.
"""

import yaml
import torch
from src.data_loading import get_dataloaders

def test_pipeline():
    """Test complet du pipeline."""
    
    print("="*60)
    print("TEST COMPLET DU PIPELINE")
    print("="*60)
    
    # Charger la config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Créer les DataLoaders
    print("\n1. Création des DataLoaders...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    print(f"✓ DataLoaders créés")
    print(f"  Meta: {meta}")
    
    # 2. Tester un batch train
    print("\n2. Test d'un batch TRAIN (avec augmentation)...")
    train_iter = iter(train_loader)
    train_images, train_labels = next(train_iter)
    
    print(f"  Shape images: {train_images.shape}")
    print(f"  Shape labels: {train_labels.shape}")
    print(f"  Type images: {train_images.dtype}")
    print(f"  Type labels: {train_labels.dtype}")
    
    # Vérifier la plage de valeurs (après normalisation)
    img_min = train_images.min().item()
    img_max = train_images.max().item()
    img_mean = train_images.mean().item()
    img_std = train_images.std().item()
    
    print(f"\n  Statistiques des images:")
    print(f"    Min: {img_min:.3f}")
    print(f"    Max: {img_max:.3f}")
    print(f"    Mean: {img_mean:.3f}")
    print(f"    Std: {img_std:.3f}")
    
    # Vérifier la normalisation
    print(f"\n  Vérification normalisation:")
    if img_min < -1.0 and img_max > 1.0:
        print(f"    ✓ Normalisation appliquée (plage attendue: ~[-2.0, 2.0])")
        print(f"    ✓ Plage actuelle: [{img_min:.3f}, {img_max:.3f}]")
    elif img_min >= 0.0 and img_max <= 1.0:
        print(f"    ✗ Normalisation NON appliquée (plage [0.0, 1.0])")
    else:
        print(f"    ? Plage inattendue: [{img_min:.3f}, {img_max:.3f}]")
    
    # Vérifier les labels
    print(f"\n  Labels:")
    print(f"    Plage: [{train_labels.min().item()}, {train_labels.max().item()}]")
    print(f"    Exemples: {train_labels[:5].tolist()}")
    
    # 3. Tester un batch validation (sans augmentation)
    print("\n3. Test d'un batch VALIDATION (sans augmentation)...")
    val_iter = iter(val_loader)
    val_images, val_labels = next(val_iter)
    
    print(f"  Shape images: {val_images.shape}")
    print(f"  Shape labels: {val_labels.shape}")
    
    val_img_min = val_images.min().item()
    val_img_max = val_images.max().item()
    print(f"  Plage valeurs: [{val_img_min:.3f}, {val_img_max:.3f}]")
    
    # 4. Comparer train vs val (pour voir l'effet des augmentations)
    print("\n4. Comparaison Train vs Validation...")
    print(f"  Train - plage: [{img_min:.3f}, {img_max:.3f}]")
    print(f"  Val   - plage: [{val_img_min:.3f}, {val_img_max:.3f}]")
    
    # Les deux devraient avoir la même plage (normalisation identique)
    if abs(img_min - val_img_min) < 0.1 and abs(img_max - val_img_max) < 0.1:
        print(f"  ✓ Plages similaires (normalisation identique)")
    else:
        print(f"  ? Plages différentes (peut être dû aux augmentations)")
    
    # 5. Vérifier les formes
    print("\n5. Vérification des formes...")
    expected_shape = (config['train']['batch_size'], *meta['input_shape'])
    
    if train_images.shape == expected_shape:
        print(f"  ✓ Train shape correct: {train_images.shape} == {expected_shape}")
    else:
        print(f"  ✗ Train shape incorrect: {train_images.shape} != {expected_shape}")
    
    if val_images.shape == expected_shape:
        print(f"  ✓ Val shape correct: {val_images.shape} == {expected_shape}")
    else:
        print(f"  ✗ Val shape incorrect: {val_images.shape} != {expected_shape}")
    
    # 6. Test de plusieurs batchs pour vérifier le shuffle
    print("\n6. Vérification du shuffle (train)...")
    train_iter = iter(train_loader)
    batch1_labels = next(train_iter)[1]
    batch2_labels = next(train_iter)[1]
    
    if not torch.equal(batch1_labels, batch2_labels):
        print(f"  ✓ Shuffle actif (labels différents entre batchs)")
        print(f"    Batch 1 (premiers labels): {batch1_labels[:5].tolist()}")
        print(f"    Batch 2 (premiers labels): {batch2_labels[:5].tolist()}")
    else:
        print(f"  ✗ Shuffle inactif (mêmes labels)")
    
    # 7. Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    
    all_ok = True
    
    # Vérifications
    checks = [
        ("Formes correctes", train_images.shape == expected_shape),
        ("Normalisation appliquée", img_min < -1.0 and img_max > 1.0),
        ("Labels dans [0, 9]", train_labels.min() >= 0 and train_labels.max() <= 9),
        ("Shuffle actif", not torch.equal(batch1_labels, batch2_labels)),
    ]
    
    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if not check_result:
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ TOUS LES TESTS PASSENT - Pipeline fonctionnel !")
    else:
        print("✗ CERTAINS TESTS ÉCHOUENT - Vérifier les erreurs ci-dessus")
    print("="*60)
    
    return all_ok

if __name__ == "__main__":
    test_pipeline()

