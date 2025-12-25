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

**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l'impact potentiel sur l'entraînement.  

La distribution des classes a été calculée en exécutant `python -m src.explore_dataset --config configs/config.yaml`. Voici le tableau de distribution :

| Classe | Train | Validation | Test | Total |
|--------|-------|------------|------|-------|
| AnnualCrop | 1 791 | 613 | 596 | 3 000 |
| Forest | 1 787 | 605 | 608 | 3 000 |
| HerbaceousVegetation | 1 799 | 628 | 573 | 3 000 |
| Highway | 1 505 | 499 | 496 | 2 500 |
| Industrial | 1 492 | 507 | 501 | 2 500 |
| Pasture | 1 195 | 409 | 396 | 2 000 |
| PermanentCrop | 1 481 | 481 | 538 | 2 500 |
| Residential | 1 863 | 583 | 554 | 3 000 |
| River | 1 460 | 511 | 529 | 2 500 |
| SeaLake | 1 827 | 564 | 609 | 3 000 |
| **TOTAL** | **16 200** | **5 400** | **5 400** | **27 000** |

**Commentaire sur l'impact pour l'entraînement** : Le dataset est relativement équilibré avec un ratio max/min de 1.56 (classe la plus fréquente : 1 863 exemples en train, classe la moins fréquente : 1 195 exemples). Cette légère différence n'est pas suffisante pour nécessiter des poids de classes ou un échantillonnage stratifié. L'entraînement standard avec une fonction de perte CrossEntropyLoss devrait fonctionner correctement sans biais significatif vers les classes majoritaires.

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).

Les particularités suivantes ont été détectées lors de l'exploration du dataset :

- ✓ **Toutes les images ont une taille uniforme** : 64×64 pixels (vérifié sur un échantillon de 100 images)
- ✓ **Images en RGB** : 3 canaux (mode RGB confirmé)
- ✓ **Pas de valeurs manquantes** : dataset complet avec 27 000 exemples étiquetés
- ✓ **Labels entiers de 0 à 9** : classification multiclasses (une seule classe par image, pas multi-label)
- ✓ **Distribution des classes relativement équilibrée** : ratio max/min = 1.56, ce qui est acceptable pour un entraînement standard

Aucune particularité problématique n'a été détectée. Le dataset est prêt pour l'entraînement sans nécessiter de prétraitements spéciaux au-delà de la normalisation standard.

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

- **Vision** :
  - `Resize` : (64, 64) — redimensionnement à 64×64 pixels
  - `ToTensor` : conversion PIL Image → tensor PyTorch (normalise [0, 255] → [0.0, 1.0])
  - `Normalize` : mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (statistiques ImageNet)

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?  

Les prétraitements suivants sont appliqués dans l'ordre suivant (via `torchvision.transforms.Compose`) :

1. **`Resize(64, 64)`** : Redimensionne toutes les images à 64×64 pixels. Bien que les images EuroSAT RGB soient déjà à cette taille (vérifié lors de l'exploration), cette étape garantit l'uniformité et permet de gérer d'éventuelles variations. Cette opération est nécessaire pour que toutes les images aient exactement la même dimension d'entrée pour le modèle.

2. **`ToTensor()`** : Convertit l'image PIL (format H×W×C, valeurs entières 0-255) en tensor PyTorch (format C×H×W, valeurs flottantes normalisées [0.0, 1.0]). Cette conversion est essentielle car PyTorch travaille avec des tensors, et la normalisation des valeurs dans [0, 1] améliore la stabilité numérique lors de l'entraînement.

3. **`Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`** : Normalise chaque canal RGB en soustrayant la moyenne et en divisant par l'écart-type. Ces valeurs correspondent aux statistiques ImageNet, couramment utilisées pour les CNNs entraînés from scratch. La normalisation centre les données autour de zéro et réduit la variance, ce qui accélère la convergence et améliore la stabilité de l'entraînement. Bien que ces statistiques ne soient pas spécifiques à EuroSAT, elles sont appropriées pour un CNN from scratch et permettent une comparaison avec d'autres travaux.

**Ordre d'application** : Resize → ToTensor → Normalize (l'ordre est important : ToTensor doit précéder Normalize car cette dernière opère sur des tensors).

**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?

Non, les prétraitements sont **identiques** pour train, validation et test. Toutes les transformations (Resize, ToTensor, Normalize) sont déterministes et appliquées de la même manière sur les trois splits. Aucun recadrage aléatoire n'est utilisé dans les prétraitements (les augmentations aléatoires sont appliquées séparément, uniquement sur le split d'entraînement, via `get_augmentation_transforms`). Cette cohérence garantit que les données de validation et de test sont évaluées dans les mêmes conditions, permettant une comparaison équitable des performances.

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - **RandomHorizontalFlip** : probabilité p=0.5 (50% de chance d'appliquer le flip)
  - **RandomCrop** : taille (64, 64), padding=4 pixels
  - **ColorJitter** : brightness=0.1 (±10%), contrast=0.1 (±10%), saturation=0.1 (±10%), hue=0.0 (pas de variation de teinte)

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  

Les augmentations suivantes sont appliquées **uniquement au split d'entraînement** dans l'ordre suivant :

1. **`RandomCrop(size=(64, 64), padding=4)`** : Recadre aléatoirement une zone de 64×64 pixels de l'image après avoir ajouté 4 pixels de padding autour. Cette augmentation ajoute de la variabilité spatiale en simulant différentes vues ou cadrages de la même scène. Le padding permet de ne pas perdre d'information en ajoutant des pixels (par répétition) autour de l'image avant le recadrage. **Justification** : Les images satellitaires peuvent être capturées sous différents angles ou avec des cadrages légèrement différents, tout en représentant la même classe d'occupation du sol.

2. **`RandomHorizontalFlip(p=0.5)`** : Retourne horizontalement l'image avec une probabilité de 50%. **Justification** : Les images satellitaires peuvent être visualisées sous différentes orientations sans changer la nature de la scène. Un champ agricole reste un champ agricole même si on le retourne horizontalement. Cette augmentation double virtuellement la taille du dataset d'entraînement et améliore la robustesse du modèle aux variations d'orientation.

3. **`ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)`** : Applique des variations aléatoires légères de couleur (±10% pour brightness, contrast, saturation, pas de variation de teinte). **Justification** : Les images satellitaires peuvent présenter des variations d'éclairage dues aux conditions météorologiques, à l'heure de la journée, à la saison, ou aux conditions atmosphériques. Ces variations ne changent pas la classe de l'image (un résidentiel reste résidentiel même avec un éclairage différent). Les paramètres sont choisis légers (±10%) pour rester réalistes et ne pas trop déformer l'apparence naturelle des images.

**Ordre d'application** : RandomCrop → RandomHorizontalFlip → ColorJitter. L'ordre est important : RandomCrop doit être appliqué en premier pour travailler sur l'image complète (avant padding), puis les autres transformations sont appliquées sur l'image recadrée.

**Impact attendu** : Ces augmentations augmentent la diversité des données d'entraînement sans changer les labels, ce qui réduit le risque de sur-apprentissage et améliore la généralisation du modèle à de nouvelles images.

**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

Oui, **toutes les augmentations appliquées sont label-preserving** (elles ne changent pas la classe de l'image). Voici la justification pour chaque transformation :

1. **RandomCrop** : Le recadrage aléatoire ne change que la zone visible de l'image, pas son contenu sémantique. Une image de "Forest" recadrée reste une forêt, une image de "Residential" recadrée reste résidentielle. Le padding de 4 pixels est suffisamment petit pour ne pas introduire d'artefacts significatifs qui changeraient la classe.

2. **RandomHorizontalFlip** : Le retournement horizontal est une transformation géométrique qui préserve la structure et le contenu de l'image. Les caractéristiques visuelles qui définissent une classe (par exemple, la texture d'une forêt, la structure d'un bâtiment résidentiel) restent reconnaissables après un flip horizontal. Cette transformation est couramment utilisée en vision par ordinateur car elle préserve les labels tout en augmentant la diversité des données.

3. **ColorJitter** : Les variations de couleur (brightness, contrast, saturation) modifient uniquement l'apparence visuelle de l'image, pas sa structure sémantique. Un champ agricole reste un champ agricole même si l'éclairage est légèrement différent. Les paramètres sont choisis légers (±10%) pour rester dans des limites réalistes qui ne changent pas fondamentalement l'apparence de la scène. La teinte (hue) n'est pas modifiée (hue=0.0) car une variation de teinte pourrait changer la perception des couleurs naturelles (par exemple, un champ vert pourrait apparaître jaune), ce qui pourrait être problématique pour la classification.

**Conclusion** : Toutes les augmentations sont conçues pour simuler des variations naturelles qui peuvent survenir dans les images satellitaires réelles, tout en préservant l'identité de la classe. Aucune transformation ne modifie la structure sémantique ou le contenu qui définit la classe d'occupation du sol.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Les exemples visuels ont été générés en exécutant `python -m src.verify_pipeline --config configs/config.yaml` et sont sauvegardés dans `artifacts/train_examples_augmented.png` et `artifacts/val_examples.png`._

**D10.** Montrez 2–3 exemples et commentez brièvement.  

Les exemples suivants illustrent les données après preprocessing et augmentation :

**Exemples train (avec augmentation)** :
- Image 1 : [À insérer depuis artifacts/train_examples_augmented.png]
- Image 2 : [À insérer depuis artifacts/train_examples_augmented.png]
- Image 3 : [À insérer depuis artifacts/train_examples_augmented.png]

**Exemples validation (sans augmentation)** :
- Image 1 : [À insérer depuis artifacts/val_examples.png]
- Image 2 : [À insérer depuis artifacts/val_examples.png]

**Commentaire** : Les images train montrent les effets des augmentations (flip horizontal, recadrage aléatoire, variations de couleur légères), tandis que les images validation sont fixes et représentent fidèlement les données originales. Les images sont correctement normalisées (valeurs dans la plage attendue après normalisation ImageNet) et ont la forme (3, 64, 64) comme attendu. Les labels correspondent bien aux classes (0-9).

**D11.** Donnez la **forme exacte** d'un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

La forme exacte d'un batch train est : `(batch_size, 3, 64, 64)` où :
- `batch_size` : nombre d'exemples par batch (configuré à 64 dans `config.yaml`)
- `3` : nombre de canaux RGB (correspond à `meta["input_shape"][0]`)
- `64` : hauteur en pixels (correspond à `meta["input_shape"][1]`)
- `64` : largeur en pixels (correspond à `meta["input_shape"][2]`)

**Vérification de cohérence** :
- `meta["input_shape"] = (3, 64, 64)` (format C, H, W)
- Forme d'un batch : `(batch_size, 3, 64, 64)` (format batch_size, C, H, W)
- **Cohérence** : ✓ OUI — Les dimensions spatiales et de canaux correspondent exactement. La seule différence est l'ajout de la dimension `batch_size` en première position, ce qui est attendu pour les batchs PyTorch.

**Vérifications supplémentaires** :
- Les labels ont la forme `(batch_size,)` avec des valeurs entières dans [0, 9]
- La plage de valeurs des images après normalisation est environ [-2.0, 2.0] (cohérent avec la normalisation ImageNet)
- Le DataLoader train a le shuffle activé (ordre aléatoire des exemples)
- Les DataLoaders validation et test ont le shuffle désactivé (ordre fixe pour reproductibilité)

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


