# DL-crossvit-multibranch-segmentation

## Description
Deep learning project implementing CrossViT  with multi-branch architecture for image segmentation tasks.

Projet de Deep learning, implementant l'architecture CrossViT (Cross-Attention Vision Transformer) multi-branche pour de la segmentation d'image.
## Features
- Architecture a base de Vision Transformer
- Croisemment de l'attention
- Pipeline multi-branche en accord avec le sujet

## Installation
```bash
git clone <repository-url>
cd DL-crossvit-multibranch-segmentation
uv venv
source .venv/bin/activate #.\.venv\Scripts\activate sous Windows
uv pip install -r requirements.txt
```
## Usage
Bien indiquer dans le fichier configs/global.yaml le chemin d'acces aux images normales et segmentees du dataset ainsi que leurs labels.

Il y a 3 mode :

- train pour entrainer le modele et generer les courbes d'apprentissage
- eval pour obtenir les metriques du modele (accuracy, recall, etc.) 
- interpret pour generer les heatmaps d'attention


Il y a 8 configuration accessibles :
- 'A' : images non segmentees
- B : images segmentees
- C1 : branche large : segmentees, branche petite : non-segmentees
- C2 : branche large : non-segmentees, branche petite : segmentees
- O2 : branches egales, images segmentees et non-segmentees
- O3 : branches egales, images segmentees et non-segmentees, ponderation par patch
- O5 : O2 avec IoU et Rollout
- O6 : O3 avec IoU et Rollout

Commande pour lancer le programme :
```bash
uv run test.py --config configs/global.yaml --mode MODE --config_name CONFIG 
```

## Project Structure
```
├── configs/
├── models/
├── src/
├── vendor/
└── README.md
```

