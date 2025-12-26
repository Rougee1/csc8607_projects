"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import os
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """
    Initialise les seeds pour numpy, torch, random et CUDA.
    
    Args:
        seed: valeur de la seed à utiliser
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Pour CUDA (si disponible)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Pour la reproductibilité des opérations déterministes
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = "auto") -> str:
    """
    Retourne 'cpu' ou 'cuda' selon la disponibilité et la préférence.
    
    Args:
        prefer: "auto" (détection automatique), "cuda" (force CUDA), "cpu" (force CPU)
    
    Returns:
        str: "cpu" ou "cuda"
    """
    if prefer == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("Warning: CUDA demandé mais non disponible, utilisation de CPU")
            return "cpu"
    elif prefer == "cpu":
        return "cpu"
    else:  # "auto"
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


def count_parameters(model) -> int:
    """
    Retourne le nombre de paramètres entraînables du modèle.
    
    Args:
        model: modèle PyTorch (nn.Module)
    
    Returns:
        int: nombre total de paramètres entraînables
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """
    Sauvegarde une copie de la config (YAML) dans out_dir.
    
    Args:
        config: dictionnaire de configuration
        out_dir: répertoire de sortie (sera créé si nécessaire)
    """
    # Créer le répertoire si nécessaire
    os.makedirs(out_dir, exist_ok=True)
    
    # Chemin du fichier de sortie
    config_path = os.path.join(out_dir, "config_snapshot.yaml")
    
    # Sauvegarder en YAML
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config sauvegardée dans: {config_path}")
