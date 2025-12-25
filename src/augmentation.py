"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)

Les augmentations sont appliquées UNIQUEMENT au split d'entraînement.
Elles doivent être label-preserving (ne pas changer la classe de l'image).
"""

import torchvision.transforms as transforms


def get_augmentation_transforms(config: dict):
    """
    Retourne les transformations d'augmentation pour les images d'entraînement.
    
    Pour EuroSAT RGB (images satellitaires), les augmentations incluent :
    - Flip horizontal : simule différentes orientations de vue
    - Random crop avec padding : ajoute de la variabilité spatiale
    - Color jitter léger : simule différentes conditions d'éclairage/atmosphériques
    
    Toutes ces transformations sont label-preserving (ne changent pas la classe).
    
    Args:
        config: Dictionnaire de configuration (section augment)
    
    Returns:
        Composition de transformations PyTorch (torchvision.transforms.Compose)
        ou None si aucune augmentation n'est configurée
    """
    augment_config = config.get('augment', {})
    
    # Liste des transformations d'augmentation
    transform_list = []
    
    # 1. Random Horizontal Flip (flip horizontal aléatoire)
    # Pourquoi : Les images satellitaires peuvent être vues sous différents angles.
    # Un champ agricole reste un champ agricole même si on le retourne horizontalement.
    # Probabilité par défaut : 0.5 (50% de chance de flip)
    if augment_config.get('random_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # 2. Random Crop avec padding (recadrage aléatoire)
    # Pourquoi : Ajoute de la variabilité spatiale en recadrant différentes parties de l'image.
    # Le padding permet de ne pas perdre d'information en ajoutant des pixels autour.
    # Utile pour que le modèle apprenne à reconnaître les classes même si la vue est légèrement décalée.
    random_crop_config = augment_config.get('random_crop')
    if random_crop_config is not None:
        crop_size = random_crop_config.get('size', [64, 64])
        padding = random_crop_config.get('padding', 4)
        
        # Convertir en tuple si c'est une liste
        if isinstance(crop_size, list):
            crop_size = tuple(crop_size)
        elif isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        
        # RandomCrop : prend une zone aléatoire de l'image (après padding)
        # Le padding ajoute des pixels autour de l'image avant le crop
        transform_list.append(transforms.RandomCrop(size=crop_size, padding=padding))
    
    # 3. Color Jitter (variation de couleur légère)
    # Pourquoi : Les images satellitaires peuvent avoir des variations d'éclairage,
    # de conditions atmosphériques, ou de saison. Le color jitter simule ces variations.
    # Paramètres légers pour ne pas trop déformer l'image (restent réalistes).
    color_jitter_config = augment_config.get('color_jitter')
    if color_jitter_config is not None:
        brightness = color_jitter_config.get('brightness', 0.1)  # Variation de luminosité ±10%
        contrast = color_jitter_config.get('contrast', 0.1)       # Variation de contraste ±10%
        saturation = color_jitter_config.get('saturation', 0.1) # Variation de saturation ±10%
        hue = color_jitter_config.get('hue', 0.0)                # Pas de variation de teinte (0.0)
        
        # ColorJitter : applique des variations aléatoires de couleur
        # Les valeurs sont des facteurs (0.1 = ±10% de variation)
        transform_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
        )
    
    # Si aucune augmentation n'est configurée, retourner None
    # (les augmentations ne seront pas appliquées)
    if len(transform_list) == 0:
        return None
    
    # Compose toutes les transformations en une seule chaîne
    # L'ordre d'application : RandomCrop → RandomHorizontalFlip → ColorJitter
    # Note : RandomCrop doit être avant les autres pour travailler sur l'image complète
    return transforms.Compose(transform_list)