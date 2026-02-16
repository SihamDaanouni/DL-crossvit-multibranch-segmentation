# DL-crossvit-multibranch-segmentation

## Description
Projet de Deep Learning implémentant l'architecture CrossViT (Cross-Attention Vision Transformer) multi-branche pour la segmentation d'images.

## Fonctionnalités
- Architecture à base de Vision Transformer
- Mécanisme de cross-attention
- Pipeline multi-branche conforme au sujet

## Installation
```bash
git clone <repository-url>
cd DL-crossvit-multibranch-segmentation
uv venv
source .venv/bin/activate  # .venv\Scripts\activate sous Windows
uv pip install -r requirements.txt
```

## Utilisation
N'oubliez pas d'indiquer dans le fichier `configs/global.yaml` les chemins d'accès aux images normales et segmentées du dataset, ainsi que leurs labels.

Il y a 3 modes disponibles :
- `train` : pour entraîner le modèle et générer les courbes d'apprentissage
- `eval` : pour obtenir les métriques du modèle (accuracy, recall, etc.)
- `interpret` : pour générer les heatmaps d'attention

Il y a 8 configurations accessibles :
- `A` : images non segmentées
- `B` : images segmentées
- `C1` : branche large : segmentées, branche petite : non segmentées
- `C2` : branche large : non segmentées, branche petite : segmentées
- `O2` : branches égales, images segmentées et non segmentées
- `O3` : branches égales, images segmentées et non segmentées, pondération par patch
- `O5` : O2 avec IoU et Rollout
- `O6` : O3 avec IoU et Rollout

Commande pour lancer le programme :
```bash
uv run test.py --config configs/global.yaml --mode MODE --config_name CONFIG
```

## Structure du projet
```
├── configs/
├── models/
├── src/
├── vendor/
└── README.md
```
## Conclusion

Ce projet a permis d'explorer des architectures avancées de Vision Transformers pour l'analyse d'herbiers. Nous avons montré que si l'architecture CrossViT de base est performante, son adaptation spécifique aux données segmentées via un routage intelligent (C1/C2) et une régularisation par l'attention (Loss IoU) permet d'améliorer significativement à la fois les métriques de classification et l'explicabilité du modèle.

L'introduction de la perte IoU sur les rollouts d'attention s'avère être une technique prometteuse pour aligner les représentations latentes des réseaux de neurones avec des connaissances a priori (la segmentation), sans nécessiter une architecture de segmentation dédiée complète.
