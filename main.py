import hydra
from omegaconf import DictConfig
from runner import run_experiment

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    run_experiment(cfg)

if __name__ == "__main__":
    main()
