import os
import sys
import hydra
import math
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from typing import Dict, List, Any

from utils.wandb_utils import init_wandb, log_system_info, log_image_with_prediction

from .training import train_model, evaluate_model

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run(cfg: DictConfig):
    """
    Main entry point for the radiology_peft experiment
    
    Args:
        cfg: Hydra configuration
    """
    # Set random seed
    set_seed(cfg.seed)
    
    # Initialize wandb if enabled
    if cfg.wandb.mode != "disabled":
        run = init_wandb(cfg)
        log_system_info()
    
    # Log GPU information
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}, Total Memory: {max_memory} GB, Start Memory: {start_gpu_memory} GB")
    
    # Prepare datasets
    print("Loading datasets...")
    train_data, eval_data, test_data = prepare_datasets(cfg)
    
    # Train the model
    print(f"Starting training with {len(train_data)} examples...")
    model, tokenizer, training_stats = train_model(cfg, train_data, eval_data)
    
    # Log final stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        print(f"Used Memory: {used_memory} GB, Used Memory for LoRA: {used_memory_for_lora} GB")
    
    # Run evaluation on test set
    print(f"Running final evaluation on {len(test_data)} test examples...")
    eval_results = evaluate_model(cfg, model, tokenizer, test_data)
    
    # Save model if needed
    if cfg.training.logging.save_model_checkpoint:
        save_path = os.path.join(cfg.output_dir, "model")
        print(f"Saving model to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    print("Experiment completed successfully!")
    return eval_results

def prepare_datasets(cfg: DictConfig):
    """
    Prepare training, validation and test datasets
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Tuple of train, validation and test datasets
    """
    dataset = load_dataset(cfg.dataset.name)
    
    train_dataset = dataset[cfg.dataset.train_split].select(
        range(cfg.dataset.start_train_idx, cfg.dataset.start_train_idx + cfg.dataset.train_size)
    )
    
    eval_dataset = dataset[cfg.dataset.train_split].select(
        range(cfg.dataset.start_eval_idx, cfg.dataset.start_eval_idx + cfg.dataset.eval_size)
    )
    
    test_dataset = dataset[cfg.dataset.eval_split].select(
        range(cfg.dataset.test_size)
    )
    
    # Convert to conversation format
    train_data = [convert_to_conversation(sample, cfg.dataset.instruction) for sample in train_dataset]
    eval_data = [convert_to_conversation(sample, cfg.dataset.instruction) for sample in eval_dataset]
    
    return train_data, eval_data, test_dataset

def convert_to_conversation(sample: Dict[str, Any], instruction: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert a dataset sample to conversation format for the model
    
    Args:
        sample: Dataset sample
        instruction: Instruction prompt
        
    Returns:
        Conversation formatted sample
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["caption"]}
            ]
        },
    ]
    return {"messages": conversation}

if __name__ == "__main__":
    # Allow running this module directly for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.append(project_root)
    
    @hydra.main(config_path="../../configs", config_name="config")
    def main(cfg: DictConfig):
        return run(cfg)
    
    main()