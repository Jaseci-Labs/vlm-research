from omegaconf import DictConfig
from .logic import VLMFinetuneExperiment
import wandb

def run(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project, config=dict(cfg), group=cfg.experiment, entity=cfg.wandb.team)
    experiment = VLMFinetuneExperiment(cfg)
    experiment.train()
    wandb.finish()