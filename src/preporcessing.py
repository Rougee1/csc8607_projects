"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

import torchvision.transforms as transforms


def get_preprocess_transforms(config: dict):
    """
    Retourne les transformations de pré-traitement pour les images.
    
    Pour EuroSAT RGB (vision), les transformations incluent :
    - Conversion en tensor PyTorch
    - Redimensionnement (si nécessaire)
    - Normalisation par moyenne/écart-type
    
    Ces transformations sont appliquées de manière identique à train/val/test.
    
    Args:
        config: Dictionnaire de configuration (section preprocess)
    
    Returns:
        Composition de transformations PyTorch (torchvision.transforms.Compose)
    """
    preprocess_config = config.get('preprocess', {})
    
    # Liste des transformations à appliquer
    transform_list = []
    
    # 1. Redimensionnement (si spécifié dans la config)
    resize = preprocess_config.get('resize')
    if resize is not None:
        # resize peut être [64, 64] ou un entier
        if isinstance(resize, list):
            target_size = tuple(resize)  # [64, 64] -> (64, 64)
        else:
            target_size = (resize, resize)  # 64 -> (64, 64)
        
        transform_list.append(transforms.Resize(target_size))
    
    # 2. Conversion en tensor PyTorch (doit être avant la normalisation)
    # Convertit PIL Image (H, W, C) en tensor (C, H, W) et normalise les valeurs [0, 255] -> [0.0, 1.0]
    transform_list.append(transforms.ToTensor())
    
    # 3. Normalisation par moyenne/écart-type
    normalize_config = preprocess_config.get('normalize')
    if normalize_config is not None:
        mean = normalize_config.get('mean', [0.485, 0.456, 0.406])  # ImageNet par défaut
        std = normalize_config.get('std', [0.229, 0.224, 0.225])   # ImageNet par défaut
        
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    # Compose toutes les transformations en une seule chaîne
    # Les transformations sont appliquées dans l'ordre : Resize -> ToTensor -> Normalize
    return transforms.Compose(transform_list)