import hydra
from omegaconf import DictConfig
import yaml
import importlib
import os

def load_registry(registry_path: str):
    # Use absolute path relative to runner.py
    abs_path = os.path.join(os.path.dirname(__file__), '..', 'experiment_registry.yaml')
    with open(abs_path, 'r') as f:
        registry = yaml.safe_load(f)
    return registry

def run_experiment(cfg: DictConfig):
    registry = load_registry('experiment_registry.yaml')
    experiment_name = cfg.experiment
    
    for exp in registry['experiments']:
        if exp['name'] == experiment_name and exp['active']:
            module_path = exp['module']
            module_name, func_name = module_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            return func(cfg)
    
    raise ValueError(f"Experiment '{experiment_name}' not found or inactive in registry")