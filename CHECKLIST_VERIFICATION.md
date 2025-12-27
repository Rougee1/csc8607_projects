# Checklist de Vérification — Modèle & Hyperparamètres

## ✅ Vérification complète de la checklist

### ✅ 1. Le modèle compile et les shapes sont correctes

**Statut :** ✅ **COMPLÉTÉ**

- **Vérification** : Section M1 et M2 du rapport
- **Shapes vérifiées** :
  - Entrée : `(batch_size, 3, 64, 64)` ✓
  - Sortie (logits) : `(batch_size, 10)` ✓
  - Labels : `(batch_size,)` ✓
- **Loss adaptée** : `CrossEntropyLoss` pour classification multiclasses ✓
- **Nombre de paramètres** : Documenté (1.15M pour blocks=2, 1.93M pour blocks=3)

**Références dans le rapport :**
- Section M1 : Architecture complète et nombre de paramètres
- Section M2 : Formes des données vérifiées

---

### ✅ 2. La loss initiale est cohérente (et débuggée sur un batch)

**Statut :** ✅ **COMPLÉTÉ**

- **Loss initiale observée** : `2.344882`
- **Loss théorique** : `-log(1/10) = 2.302585`
- **Différence** : `0.042297` (< 0.05, cohérent) ✓
- **Gradients vérifiés** : Norme totale = `3.478104`, non-nuls ✓
- **Backward OK** : Rétropropagation fonctionnelle ✓

**Références dans le rapport :**
- Section M2 : Perte initiale et vérification du premier batch
- Script utilisé : `python -m src.check_initial_loss --config configs/config.yaml`

---

### ✅ 3. Overfit small obtenu (train/loss ↓ vers 0 sur un mini-subset)

**Statut :** ✅ **COMPLÉTÉ**

- **Taille du sous-ensemble** : 32 exemples
- **Hyperparamètres** : `blocks_per_stage=2`, `dilation_stage3=2`
- **LR** : 0.01 (élevé pour mémorisation rapide)
- **Weight decay** : 0.0 (désactivé)
- **Résultat** : Loss descend de `2.343865` → `0.009035` en 33 époques ✓
- **Graphique** : `artifacts/overfit_small_loss_32ex.png` ✓

**Références dans le rapport :**
- Section M3 : Overfit sur petit échantillon
- Script utilisé : `python -m src.overfit_small --config configs/config.yaml --overfit_size 32 --epochs 50 --lr 0.01`

---

### ✅ 4. LR choisi via LR finder, grid rapide effectuée

**Statut :** ✅ **COMPLÉTÉ**

**LR Finder :**
- **Méthode** : Balayage logarithmique (1e-5 à 1e-1, 50 valeurs)
- **LR recommandé** : `0.000910` (9.10e-04)
- **Weight decay choisi** : `1e-4` (puis ajusté à `1e-5` après grid search)
- **Graphiques** : 4 graphiques dans `artifacts/lr_finder_*.png` ✓

**Grid Search :**
- **Grille** : LR `{0.0005, 0.001, 0.002}`, WD `{1e-5, 1e-4}`, Dilation `{2, 3}`, Blocks `{2, 3}`
- **Total** : 24 combinaisons
- **Époques par run** : 5
- **Meilleure combinaison** : LR=0.0005, WD=1e-5, Dilation=2, Blocks=3 (Val Acc: 90.52%)
- **Graphiques** : `artifacts/grid_search_*.png` ✓
- **CSV** : `artifacts/grid_search_results.csv` ✓

**Références dans le rapport :**
- Section M4 : LR Finder - Choix du Learning Rate
- Section M5 : Mini Grid Search - Résultats et Analyse
- Scripts utilisés :
  - `python -m src.lr_finder --config configs/config.yaml`
  - `python -m src.grid_search --config configs/config.yaml --epochs 5`

---

### ✅ 5. Entraînement 10-20 époques sur la meilleure config, best.ckpt sauvegardé

**Statut :** ✅ **COMPLÉTÉ**

- **Configuration finale** : LR=0.0003, WD=1e-5, Dilation=2, Blocks=3
- **Époques** : 20
- **Meilleure Val Accuracy** : 96.52% (epoch 17)
- **Checkpoint sauvegardé** : `artifacts/best.ckpt` ✓
- **Graphiques** : `artifacts/training_curves.png`, `artifacts/training_curves_comparison.png` ✓
- **Logs TensorBoard** : `runs/train_lr=0.0003_wd=1e-05_dil=2_blk=3_*` ✓

**Références dans le rapport :**
- Section M6 : Entraînement complet - Courbes d'apprentissage
- Script utilisé : `python -m src.train --config configs/config.yaml --max_epochs 20`

---

### ✅ 6. Courbes comparatives claires dans TensorBoard + captures dans le rapport

**Statut :** ✅ **COMPLÉTÉ**

**Comparaisons générées :**
1. **Comparaison LR** : `artifacts/comparison_lr.png` ✓
2. **Comparaison Weight Decay** : `artifacts/comparison_weight_decay.png` ✓
3. **Comparaison Hyperparamètres Modèle** : `artifacts/comparison_model_hparams.png` ✓

**Analyse dans le rapport :**
- Section M7 : Comparaisons de courbes - Analyse des hyperparamètres
- Chaque comparaison inclut : Attendu vs Observé
- Graphiques intégrés avec légendes complètes ✓

**Script utilisé :**
- `python -m src.compare_curves --config configs/config.yaml`

---

### ✅ 7. Évaluation finale sur test rapportée et interprétée

**Statut :** ✅ **COMPLÉTÉ**

- **Checkpoint évalué** : `artifacts/best.ckpt` ✓
- **Test Accuracy** : **97.02%** ✓
- **Validation Accuracy** : 96.52%
- **Écart** : +0.50% (excellente généralisation) ✓
- **Matrice de confusion** : `artifacts/confusion_matrix_test.png` ✓
- **Rapport de classification** : Par classe (precision, recall, F1-score) ✓
- **Interprétation** : Écart très faible, pas de sur-apprentissage ✓

**Références dans le rapport :**
- Section M9 : Évaluation finale sur le test set
- Script utilisé : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 📊 Résumé global

| Point | Statut | Section Rapport | Artifacts |
|-------|--------|----------------|------------|
| 1. Modèle compile & shapes | ✅ | M1, M2 | - |
| 2. Loss initiale cohérente | ✅ | M2 | - |
| 3. Overfit small obtenu | ✅ | M3 | `overfit_small_loss_32ex.png` |
| 4. LR finder + grid search | ✅ | M4, M5 | `lr_finder_*.png`, `grid_search_*.png` |
| 5. Entraînement complet | ✅ | M6 | `training_curves*.png`, `best.ckpt` |
| 6. Courbes comparatives | ✅ | M7 | `comparison_*.png` |
| 7. Évaluation finale | ✅ | M9 | `confusion_matrix_test.png` |

**Tous les points de la checklist sont complétés et documentés dans le rapport.**

---

## 📁 Artifacts vérifiés

- ✅ `artifacts/best.ckpt` : Présent
- ✅ `artifacts/training_curves*.png` : Présents
- ✅ `artifacts/comparison_*.png` : Présents (3 fichiers)
- ✅ `artifacts/grid_search_*.png` : Présents
- ✅ `artifacts/lr_finder_*.png` : Présents (4 fichiers)
- ✅ `artifacts/confusion_matrix_test.png` : Présent
- ✅ `artifacts/overfit_small_loss_32ex.png` : Présent

---

## ✅ Checklist complète — Projet prêt pour rendu

