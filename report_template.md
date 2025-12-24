# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : _Nom, Prénom_
- **Projet** : _Intitulé (dataset × modèle)_
- **Dépôt Git** : _URL publique_
- **Environnement** : `python == ...`, `torch == ...`, `cuda == ...`  
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset
- **Source** (lien) : https://huggingface.co/datasets/timm/eurosat-rgb
- **Type d’entrée** (image / texte / audio / séries) : image
- **Tâche** (multiclasses, multi-label, régression) : multiclasses
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : (3, 64, 64)
- **Nombre de classes** (`meta["num_classes"]`) : 10

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

Nous utilisons le dataset **EuroSAT RGB** (version RGB du dataset EuroSAT), disponible sur HuggingFace sous le nom `timm/eurosat-rgb`. Ce dataset provient d'images satellitaires Sentinel-2 et contient 27 000 échantillons géoréférencés et étiquetés représentant 10 classes d'occupation du sol. Le format des données est le suivant :
- **Type d'entrée** : Images RGB (3 canaux)
- **Dimensions** : 64×64 pixels par image
- **Format de stockage** : Images JPEG encodées en RGB (bandes visibles uniquement)
- **Source originale** : https://github.com/phelber/eurosat (Helber et al., 2019, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing)

### 1.2 Splits et statistiques

| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
|------:|----------:|--------------------------------------------------------|
| Train | 16 200     | Split d'entraînement (~60% du dataset total)           |
| Val   | 5 400      | Split de validation (~20% du dataset total)            |
| Test  | 5 400      | Split de test (~20% du dataset total)                  |

**D2.** Donnez la taille de chaque split et le nombre de classes.  

Le dataset EuroSAT RGB contient **27 000 exemples** au total, répartis en trois splits :
- **Train** : 16 200 exemples (~60%)
- **Validation** : 5 400 exemples (~20%)
- **Test** : 5 400 exemples (~20%)

Le nombre de classes est **10** (classification multiclasses). Les métadonnées retournées par `get_dataloaders` sont :
```python
meta = {
    "num_classes": 10,
    "input_shape": (3, 64, 64)  # (canaux, hauteur, largeur)
}
```

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).

Aucun split n'a été créé manuellement. Le dataset `timm/eurosat-rgb` sur HuggingFace fournit déjà les trois splits (train, validation, test) pré-définis. Ces splits ont été établis selon les définitions du projet Google Research (référence : https://github.com/google-research/google-research/blob/master/remote_sensing_representations/README.md#dataset-splits). Nous utilisons directement ces splits sans modification, garantissant ainsi la reproductibilité et la comparabilité avec d'autres travaux utilisant le même dataset.

**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement.  

La distribution des classes a été calculée en exécutant `python -m src.explore_dataset --config configs/config.yaml`. Voici le tableau de distribution :

| Classe | Train | Validation | Test | Total |
|--------|-------|------------|------|-------|
| AnnualCrop | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| Forest | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| HerbaceousVegetation | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| Highway | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| Industrial | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| Pasture | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| PermanentCrop | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| Residential | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| River | [à compléter] | [à compléter] | [à compléter] | [à compléter] |
| SeaLake | [à compléter] | [à compléter] | [à compléter] | [à compléter] |

**Commentaire sur l'impact pour l'entraînement** : [À compléter après exécution du script. Indiquer si le dataset est équilibré ou déséquilibré, et les implications : si déséquilibré, mentionner l'utilisation potentielle de poids de classes ou d'un échantillonnage stratifié. Si équilibré, noter que l'entraînement standard devrait fonctionner correctement.]

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).

[À compléter après exécution du script. Mentionner :]
- Toutes les images ont une taille uniforme (64×64 pixels) ou non
- Images en RGB (3 canaux)
- Pas de valeurs manquantes
- Labels entiers de 0 à 9 (classification multiclasses, pas multi-label)
- Équilibrage ou déséquilibrage des classes

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

- Vision : resize = __, center-crop = __, normalize = (mean=__, std=__)…
- Audio : resample = __ Hz, mel-spectrogram (n_mels=__, n_fft=__, hop_length=__), AmplitudeToDB…
- NLP : tokenizer = __, vocab = __, max_length = __, padding/truncation = __…
- Séries : normalisation par canal, fenêtrage = __…

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?  
**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - ex. Flip horizontal p=0.5, RandomResizedCrop scale=__, ratio=__ …
  - Audio : time/freq masking (taille, nb masques) …
  - Séries : jitter amplitude=__, scaling=__ …

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  
**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.  
**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `_____` → score = `_____`
- **Prédiction aléatoire uniforme** — Métrique : `_____` → score = `_____`  
_Commentez en 2 lignes ce que ces chiffres impliquent._

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = __(batch_size, num_classes)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).


### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `_____`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = ____` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `_____`, `_____`
- **Optimisation** : LR = `_____`, weight decay = `_____` (0 ou très faible recommandé)
- **Nombre d’époques** : `_____`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `_____ → _____`
- **Choix pour la suite** :
  - **LR** = `_____`
  - **Weight decay** = `_____` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

---

## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : `{_____ , _____ , _____}`
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A : `{_____, _____}`
  - Hyperparamètre modèle B : `{_____, _____}`

- **Durée des runs** : `_____` époques par run (1–5 selon dataset), même seed

| Run (nom explicite) | LR    | WD     | Hyp-A | Hyp-B | Val metric (nom=_____) | Val loss | Notes |
|---------------------|-------|--------|-------|-------|-------------------------|----------|-------|
|                     |       |        |       |       |                         |          |       |
|                     |       |        |       |       |                         |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `_____`
  - Weight decay = `_____`
  - Hyperparamètre modèle A = `_____`
  - Hyperparamètre modèle B = `_____`
  - Batch size = `_____`
  - Époques = `_____` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
````

* **Artifacts requis présents** :

  * [ ] `runs/` (runs utiles uniquement)
  * [ ] `artifacts/best.ckpt`
  * [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

* PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
* Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
* Toute ressource externe substantielle (une ligne par référence).


