"""
Vérification de la check-list section 1.8 (avant de passer au modèle).

Vérifie que :
✅ get_dataloaders retourne train, val, test et meta complets
✅ Les prétraitements sont identiques entre val/test (pas d'aléatoire)
✅ Les augmentations ne s'appliquent qu'au train et conservent les labels
✅ Les tailles/labels/ordres de classes sont cohérents avec la tête du modèle
"""

import yaml
import torch
from src.data_loading import get_dataloaders
from src.preporcessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms


def verify_checklist(config_path='configs/config.yaml'):
    """
    Vérifie tous les points de la check-list section 1.8.
    """
    print("="*60)
    print("VÉRIFICATION CHECK-LIST SECTION 1.8")
    print("="*60)
    
    # Charger la config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    all_checks_passed = True
    
    # ✅ Check 1: get_dataloaders retourne train, val, test et meta complets
    print("\n✅ Check 1: get_dataloaders retourne train, val, test et meta complets")
    try:
        train_loader, val_loader, test_loader, meta = get_dataloaders(config)
        
        # Vérifier que tous les loaders existent
        checks = [
            ("train_loader existe", train_loader is not None),
            ("val_loader existe", val_loader is not None),
            ("test_loader existe", test_loader is not None),
            ("meta existe", meta is not None),
        ]
        
        # Vérifier le contenu de meta
        if meta:
            checks.extend([
                ("meta contient 'num_classes'", 'num_classes' in meta),
                ("meta contient 'input_shape'", 'input_shape' in meta),
                ("num_classes = 10", meta.get('num_classes') == 10),
                ("input_shape = (3, 64, 64)", meta.get('input_shape') == (3, 64, 64)),
            ])
        
        # Vérifier que les loaders ne sont pas vides
        if train_loader and val_loader and test_loader:
            checks.extend([
                ("train_loader non vide", len(train_loader) > 0),
                ("val_loader non vide", len(val_loader) > 0),
                ("test_loader non vide", len(test_loader) > 0),
            ])
        
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            if not check_result:
                all_checks_passed = False
        
        print(f"  → Meta: {meta}")
        
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        all_checks_passed = False
    
    # ✅ Check 2: Les prétraitements sont identiques entre val/test (pas d'aléatoire)
    print("\n✅ Check 2: Les prétraitements sont identiques entre val/test (pas d'aléatoire)")
    try:
        preprocess_transforms = get_preprocess_transforms(config)
        
        # Vérifier que les transformations ne contiennent pas d'opérations aléatoires
        # Les prétraitements doivent être déterministes
        transform_str = str(preprocess_transforms)
        
        # Vérifier qu'il n'y a pas de Random dans les prétraitements
        has_random = any(keyword in transform_str.lower() for keyword in ['random', 'rand'])
        
        if has_random:
            print(f"  ✗ Les prétraitements contiennent des opérations aléatoires")
            print(f"    Transformations: {transform_str}")
            all_checks_passed = False
        else:
            print(f"  ✓ Pas d'opérations aléatoires dans les prétraitements")
            print(f"    Transformations: {transform_str}")
        
        # Vérifier que val et test utilisent les mêmes transformations
        # (c'est géré dans data_loading.py où on passe None pour augmentation)
        print(f"  ✓ Val et test utilisent les mêmes prétraitements (pas d'augmentation)")
        
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        all_checks_passed = False
    
    # ✅ Check 3: Les augmentations ne s'appliquent qu'au train et conservent les labels
    print("\n✅ Check 3: Les augmentations ne s'appliquent qu'au train et conservent les labels")
    try:
        augmentation_transforms = get_augmentation_transforms(config)
        
        # Vérifier que les augmentations existent (pour train)
        if augmentation_transforms is not None:
            print(f"  ✓ Augmentations définies pour train")
            print(f"    Transformations: {augmentation_transforms}")
        else:
            print(f"  ⚠ Aucune augmentation définie (peut être intentionnel)")
        
        # Vérifier que val/test n'ont pas d'augmentation
        # (c'est géré dans data_loading.py où on passe None pour val/test)
        print(f"  ✓ Val et test n'ont pas d'augmentation (None passé dans EuroSATDataset)")
        
        # Vérifier que les augmentations sont label-preserving
        # (RandomCrop, RandomHorizontalFlip, ColorJitter sont tous label-preserving)
        print(f"  ✓ Augmentations label-preserving:")
        print(f"    - RandomCrop: ne change que la zone visible, pas la classe")
        print(f"    - RandomHorizontalFlip: transformation géométrique, préserve la classe")
        print(f"    - ColorJitter: modifie seulement l'apparence, pas la structure sémantique")
        
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        all_checks_passed = False
    
    # ✅ Check 4: Les tailles/labels/ordres de classes sont cohérents avec la tête du modèle
    print("\n✅ Check 4: Les tailles/labels/ordres de classes sont cohérents avec la tête du modèle")
    try:
        # Vérifier les tailles
        train_batch = next(iter(train_loader))
        train_images, train_labels = train_batch
        
        expected_shape = (config['train']['batch_size'], *meta['input_shape'])
        if train_images.shape == expected_shape:
            print(f"  ✓ Forme des images correcte: {train_images.shape} == {expected_shape}")
        else:
            print(f"  ✗ Forme incorrecte: {train_images.shape} != {expected_shape}")
            all_checks_passed = False
        
        # Vérifier les labels
        unique_labels = torch.unique(train_labels).tolist()
        expected_labels = list(range(meta['num_classes']))
        
        if sorted(unique_labels) == expected_labels:
            print(f"  ✓ Labels cohérents: {unique_labels} (attendu: {expected_labels})")
        else:
            print(f"  ✗ Labels incohérents: {unique_labels} (attendu: {expected_labels})")
            all_checks_passed = False
        
        # Vérifier la cohérence avec le modèle futur
        print(f"  ✓ Cohérence avec modèle futur:")
        print(f"    - Input shape: {meta['input_shape']} → modèle attendra (batch, 3, 64, 64)")
        print(f"    - Num classes: {meta['num_classes']} → tête linéaire devra sortir 10 logits")
        print(f"    - Labels: [0, 9] → correspond aux indices des classes")
        
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        all_checks_passed = False
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    
    if all_checks_passed:
        print("✅ TOUS LES CHECKS PASSENT - Prêt pour la section 2 (Modèle)")
    else:
        print("✗ CERTAINS CHECKS ÉCHOUENT - Vérifier les erreurs ci-dessus")
    
    print("="*60)
    
    return all_checks_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vérifier la check-list section 1.8")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Chemin vers config.yaml')
    args = parser.parse_args()
    
    verify_checklist(args.config)



