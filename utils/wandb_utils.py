import wandb

def init_wandb(cfg):
    run = wandb.init(
        project=cfg.project_name,
        name=cfg.experiment_name or f"{cfg.model.name}_{cfg.experiment.type}",
        config=dict(cfg),
    )
    return run
