"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
from .preporcessing import get_preprocess_transforms  # Note: fichier s'appelle preporcessing.py (typo dans le dépôt)
from .augmentation import get_augmentation_transforms


class EuroSATDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper pour EuroSAT RGB depuis HuggingFace.
    Convertit les images PIL en tensors PyTorch et applique les transformations.
    """
    def __init__(self, hf_dataset, preprocess_transforms, augmentation_transforms=None):
        """
        Args:
            hf_dataset: Dataset HuggingFace (split train/validation/test)
            preprocess_transforms: Transformations de preprocessing (appliquées toujours)
            augmentation_transforms: Transformations d'augmentation (None pour val/test)
        """
        self.hf_dataset = hf_dataset
        self.preprocess = preprocess_transforms
        self.augment = augmentation_transforms
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        # Image PIL depuis HuggingFace
        image = item['image']
        # Label (entier)
        label = item['label']
        
        # IMPORTANT: Appliquer augmentation AVANT preprocessing
        # Les augmentations (RandomCrop, RandomFlip, ColorJitter) fonctionnent sur PIL Images
        # Le preprocessing (ToTensor, Normalize) convertit en tensor
        if self.augment:
            image = self.augment(image)
        
        # Appliquer preprocessing (toujours, après augmentation)
        if self.preprocess:
            image = self.preprocess(image)
        
        return image, label


def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    
    Args:
        config: Dictionnaire de configuration (section dataset, preprocess, augment)
    
    Returns:
        train_loader, val_loader, test_loader, meta
    """
    # Charger le dataset EuroSAT RGB depuis HuggingFace
    # Le dataset a déjà les splits train/validation/test
    dataset_name = config['dataset']['name']
    root = config['dataset']['root']
    
    print(f"Chargement du dataset {dataset_name} depuis HuggingFace...")
    # timm/eurosat-rgb a déjà les splits train/validation/test
    hf_datasets = load_dataset("timm/eurosat-rgb")
    
    # Récupérer les splits
    train_split = config['dataset']['split']['train']
    val_split = config['dataset']['split']['val']
    test_split = config['dataset']['split']['test']
    
    train_hf = hf_datasets[train_split]
    val_hf = hf_datasets[val_split]
    test_hf = hf_datasets[test_split]
    
    print(f"Train: {len(train_hf)} exemples")
    print(f"Validation: {len(val_hf)} exemples")
    print(f"Test: {len(test_hf)} exemples")
    
    # Obtenir les transformations
    preprocess_transforms = get_preprocess_transforms(config)
    augmentation_transforms = get_augmentation_transforms(config)
    
    # Créer les datasets PyTorch
    train_dataset = EuroSATDataset(train_hf, preprocess_transforms, augmentation_transforms)
    val_dataset = EuroSATDataset(val_hf, preprocess_transforms, None)  # Pas d'augmentation pour val
    test_dataset = EuroSATDataset(test_hf, preprocess_transforms, None)  # Pas d'augmentation pour test
    
    # Paramètres des DataLoaders
    batch_size = config['train']['batch_size']
    num_workers = config['dataset'].get('num_workers', 4)
    shuffle_train = config['dataset'].get('shuffle', True)
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True  # Accélère le transfert vers GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Pas de shuffle pour val/test
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Métadonnées
    # EuroSAT RGB : 10 classes, images 64x64 RGB (3 canaux)
    meta = {
        "num_classes": 10,
        "input_shape": (3, 64, 64)  # (C, H, W)
    }
    
    print(f"Métadonnées: {meta}")
    
    return train_loader, val_loader, test_loader, meta