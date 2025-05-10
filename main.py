import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.registry import load_experiment

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for all experiments.
    Loads the appropriate experiment module and runs it.
    
    Args:
        cfg: Hydra configuration
    """
    print(f"Running experiment: {cfg.experiment.name}")
    print(OmegaConf.to_yaml(cfg))
    
    # Load and run the experiment
    experiment_module = load_experiment(cfg.experiment.name)
    experiment_module.run(cfg)

if __name__ == "__main__":
    main()