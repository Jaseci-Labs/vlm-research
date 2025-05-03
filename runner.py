from train import run_training
from evaluate import run_prompt_eval, run_data_impact
from omegaconf import DictConfig
from utils.wandb_utils import init_wandb

def run_experiment(cfg: DictConfig):
    run = None
    if cfg.use_wandb:
        run = init_wandb(cfg)

    match cfg.experiment.type:
        case "prompt_eval":
            run_prompt_eval(cfg)
        case "training_comparison":
            run_training(cfg)
        case "data_impact":
            run_data_impact(cfg)

    if run:
        run.finish()
