from guided_diffusion.train_util import TrainLoop
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

def main():
    # Configuration d'entraînement
    train_config = {
        'batch_size': 8,
        'lr': 1e-4,
        'model_name': 'diffusion_ffhq_10m',
        'num_epochs': 100,
        'save_interval': 1000,
        'eval_interval': 100,
        'train_data_dir': 'trainsets/my_dataset',
        
        # Paramètres de diffusion
        'noise_level': 12.75/255.0,
        'num_train_timesteps': 1000,
    }
    
    # Initialiser le modèle
    model_config = dict(
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
    )
    
    # Charger le modèle pré-entraîné
    model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults()
    )
    model.load_state_dict(torch.load('model_zoo/diffusion_ffhq_10m.pt'))

    # Créer le dataset d'entraînement
    train_dataset = CustomDataset(
        img_paths=util.get_image_paths(train_config['train_data_dir']),
        config=train_config
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'],
        shuffle=True
    )

    # Initialiser la boucle d'entraînement 
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=train_config['batch_size'],
        lr=train_config['lr'],
        log_interval=train_config['eval_interval'],
        save_interval=train_config['save_interval'],
        resume_checkpoint='model_zoo/diffusion_ffhq_10m.pt'
    )

    # Lancer l'entraînement
    trainer.run_loop()