# Fine-tuning DiffPIR - Guide d'utilisation

## Structure des fichiers

```plaintext
DiffPIR/
├── train_ddpir.py          # Script d'entraînement
├── configs/
│   └── finetune.yaml       # Configuration du fine-tuning
└── trainsets/
    └── dataset_xray_50/         # Vos images d'entraînement
```

## Fichiers rajoutés
1. train_ddpir.py

Nouveau script pour le fine-tuning du modèle DiffPIR. Principales caractéristiques :

- Chargement du modèle pré-entraîné
- Paramètres d'entraînement personnalisables
- Boucle d'entraînement optimisée

2. finetune.yaml

```Python
training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 100
  save_interval: 1000
  eval_interval: 100

model:
  model_name: "diffusion_ffhq_10m"
  pretrained_path: "model_zoo/diffusion_ffhq_10m.pt"
```

3. trainsets
Structure recommandée pour vos données :

```trainsets/
└── dataset_xray_50/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Utilisation
1. Préparation des données

- Résolution : 256x256 pixels
- Format : PNG/JPG
- Placement : trainsets/my_dataset/

2. Lancement du fine-tuning

```Python
python train_ddpir.py --config configs/finetune.yaml
```

## Recommandations importantes
### Données
- Maintenir une résolution constante (256x256)
- Utiliser des formats d'image standards (PNG/JPG)
- Assurer une distribution similaire aux données d'origine

### Entraînement
- Learning rate réduit : 1e-4 (ajustable dans config)
- Commencer par peu d'epochs (~100)
- Surveiller les métriques PSNR/LPIPS

## Suivi des résultats

Les checkpoints et logs seront sauvegardés dans :

```results/
└── finetune_runs/
    └── [timestamp]/
```