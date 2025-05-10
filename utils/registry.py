import importlib
import yaml
import os
from pathlib import Path

def load_experiment_registry():
    """
    Load the experiment registry from YAML file.
    
    Returns:
        dict: A mapping from experiment names to their module paths
    """
    registry_path = Path(__file__).parents[1] / "experiment_registry.yaml"
    with open(registry_path, 'r') as f:
        registry = yaml.safe_load(f)
    return registry['experiments']

def load_experiment(experiment_name):
    """
    Dynamically load an experiment module based on its name.
    
    Args:
        experiment_name: Name of the experiment as defined in registry
        
    Returns:
        module: The loaded experiment module
    """
    registry = load_experiment_registry()
    
    if experiment_name not in registry:
        raise ValueError(f"Experiment '{experiment_name}' not found in registry")
    
    module_path = registry[experiment_name]
    module = importlib.import_module(module_path)
    
    return module