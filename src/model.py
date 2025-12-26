"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import torch
import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    """
    Construit un CNN 3 stages avec dilatation au stage 3.
    
    Architecture:
    - Stage 1: 2-3 blocs (64 canaux, dilation=1, padding=1) → MaxPool 2×2
    - Stage 2: 2-3 blocs (128 canaux, dilation=1, padding=1) → MaxPool 2×2
    - Stage 3: 2-3 blocs (256 canaux, dilation=D, padding=D) → PAS de MaxPool
    - Tête: AdaptiveAvgPool2d(1) → Flatten → Linear(256 → num_classes)
    
    Args:
        config: dictionnaire de configuration avec section 'model'
    
    Returns:
        nn.Module: modèle PyTorch
    """
    model_config = config['model']
    
    # Paramètres du modèle
    num_classes = model_config['num_classes']
    channels = model_config['channels']  # [64, 128, 256]
    blocks_per_stage = model_config['blocks_per_stage']  # 2 ou 3
    dilation_stage3 = model_config['dilation_stage3']  # 2 ou 3
    use_batch_norm = model_config.get('batch_norm', True)
    activation = model_config.get('activation', 'relu')
    
    # Activation function
    if activation == 'relu':
        act_fn = nn.ReLU(inplace=True)
    elif activation == 'gelu':
        act_fn = nn.GELU()
    else:
        act_fn = nn.ReLU(inplace=True)
    
    layers = []
    
    # ========== STAGE 1 ==========
    # Input: (B, 3, 64, 64)
    in_channels = 3
    out_channels = channels[0]  # 64
    
    # Blocs du stage 1 (2 ou 3 blocs)
    for i in range(blocks_per_stage):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, dilation=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_fn)
        in_channels = out_channels  # Pour les blocs suivants
    
    # MaxPool après le stage 1
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # Output: (B, 64, 32, 32)
    
    # ========== STAGE 2 ==========
    # Input: (B, 64, 32, 32)
    in_channels = channels[0]  # 64
    out_channels = channels[1]  # 128
    
    # Blocs du stage 2 (2 ou 3 blocs)
    for i in range(blocks_per_stage):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, dilation=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_fn)
        in_channels = out_channels  # Pour les blocs suivants
    
    # MaxPool après le stage 2
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # Output: (B, 128, 16, 16)
    
    # ========== STAGE 3 ==========
    # Input: (B, 128, 16, 16)
    in_channels = channels[1]  # 128
    out_channels = channels[2]  # 256
    
    # Blocs du stage 3 (2 ou 3 blocs) avec DILATATION
    for i in range(blocks_per_stage):
        # ⚠️ IMPORTANT: padding = dilation pour conserver la taille
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=dilation_stage3, 
                               dilation=dilation_stage3))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_fn)
        in_channels = out_channels  # Pour les blocs suivants
    
    # PAS de MaxPool au stage 3
    # Output: (B, 256, 16, 16)
    
    # ========== TÊTE DE CLASSIFICATION ==========
    # Global Average Pooling
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    # Output: (B, 256, 1, 1)
    
    # Flatten
    layers.append(nn.Flatten())
    # Output: (B, 256)
    
    # Linear layer
    layers.append(nn.Linear(channels[2], num_classes))
    # Output: (B, num_classes) ← LOGITS
    
    # Construire le modèle
    model = nn.Sequential(*layers)
    
    return model
