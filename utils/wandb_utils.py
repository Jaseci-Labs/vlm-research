import os
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io

def init_wandb(cfg: DictConfig):
    """
    Initialize Weights & Biases with the given configuration.
    
    Args:
        cfg: Hydra configuration
    
    Returns:
        wandb run object
    """
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Start a new wandb run
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name if hasattr(cfg.wandb, 'name') else cfg.experiment.name,
        config=wandb_cfg,
        tags=cfg.wandb.tags if hasattr(cfg.wandb, 'tags') else None,
        group=cfg.wandb.group if hasattr(cfg.wandb, 'group') else None,
        job_type=cfg.wandb.job_type if hasattr(cfg.wandb, 'job_type') else None,
        mode=cfg.wandb.mode if hasattr(cfg.wandb, 'mode') else "online",
    )
    
    return run

def log_system_info():
    """Log system information to wandb"""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "gpu_name": device_props.name,
            "gpu_memory_total": round(device_props.total_memory / 1024 / 1024 / 1024, 2),
            "cuda_version": torch.version.cuda,
        }
        wandb.log({"system/gpu": gpu_info})
    
    cpu_info = {
        "torch_version": torch.__version__,
    }
    wandb.log({"system/cpu": cpu_info})

def log_model_gradients(model, step):
    """Log gradient statistics for model parameters"""
    grad_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_dict[f"gradients/{name}_mean"] = param.grad.abs().mean().item()
            grad_dict[f"gradients/{name}_std"] = param.grad.std().item()
    
    wandb.log(grad_dict, step=step)

def log_image_with_prediction(image, true_caption, pred_caption, step):
    """Log an image with true and predicted captions"""
    # Create a matplotlib figure
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {pred_caption[:50]}...\nGround Truth: {true_caption[:50]}...", 
              fontsize=10, pad=10)
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Log to wandb
    wandb.log({
        "predictions/samples": wandb.Image(buf, 
                              caption=f"Step {step}"),
        "predictions/true_text": true_caption,
        "predictions/pred_text": pred_caption
    }, step=step)
    
    plt.close()