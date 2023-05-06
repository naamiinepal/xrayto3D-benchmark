import wandb

wandb.login()


training_config = {
    "trainpaths": {"value": "configs/paths/verse19/Verse2019-DRR-full_train.csv"},
    "valpaths": {"value": "configs/paths/verse19/Verse2019-DRR-full_val.csv"},
    "loss": {"value": "DiceLoss"},
    "size": {"value": 64},
    "res": {"value": 1.5},
    "batch_size": {"value": 8},
    "gpu": {"value": 0},
    "steps": {"value": 5000},
    "lr": {"value": 0.0002},
}

sweep_parameters = {
    "feature_size": {"values": ['small', 'default', 'large']},
    "num_heads": {"values": ['small', 'progressive']},
}

sweep_config = {
    "program": "sweep/swin_unetr_trainer.py",
    "method": "grid",
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
    },
    "metric": {"name": "val/dice", "goal": "maximize"},
    "parameters": dict(sweep_parameters, **training_config),
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="swinunetr_sweep")

print(sweep_id)
