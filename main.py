import hydra
from omegaconf import DictConfig
from core.runner import run_experiment

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    run_experiment(cfg)

if __name__ == "__main__":
    main()