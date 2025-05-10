"""
Data loaders and processors for vision-language tasks
"""
import os
import torch
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image
import numpy as np
from omegaconf import DictConfig

def load_and_process_dataset(
    cfg: DictConfig, 
    split: str = "train", 
    start_idx: Optional[int] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = False
) -> Dataset:
    """
    Load and process a dataset from the Hugging Face Hub
    
    Args:
        cfg: Configuration with dataset details
        split: Dataset split to load
        start_idx: Starting index for subset selection
        max_samples: Maximum number of samples to include
        shuffle: Whether to shuffle the dataset
        
    Returns:
        Processed dataset
    """
    # Load the dataset
    dataset = load_dataset(cfg.dataset.name, split=split)
    
    # Apply selection if needed
    if start_idx is not None and max_samples is not None:
        dataset = dataset.select(range(start_idx, start_idx + max_samples))
    elif max_samples is not None:
        dataset = dataset.select(range(max_samples))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=cfg.seed)
    
    return dataset

def prepare_conversation_dataset(
    dataset: Dataset, 
    instruction: str,
    image_key: str = "image",
    caption_key: str = "caption"
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Convert a dataset to conversation format for vision-language models
    
    Args:
        dataset: Input dataset with images and captions
        instruction: System instruction for the task
        image_key: Key for image field in the dataset
        caption_key: Key for caption field in the dataset
        
    Returns:
        List of formatted conversations
    """
    conversations = []
    
    for sample in dataset:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample[image_key]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample[caption_key]}
                ]
            },
        ]
        conversations.append({"messages": conversation})
    
    return conversations

def convert_to_vision_dataset(
    image_paths: List[str],
    captions: Optional[List[str]] = None,
    image_transform = None
) -> Dataset:
    """
    Create a dataset from a list of image paths and optional captions
    
    Args:
        image_paths: List of paths to images
        captions: Optional list of captions (same length as image_paths)
        image_transform: Optional transform to apply to images
        
    Returns:
        Dataset with images and captions
    """
    data = []
    
    for i, img_path in enumerate(image_paths):
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
            if image_transform:
                image = image_transform(image)
                
            # Create sample
            sample = {"image": image}
            
            # Add caption if available
            if captions is not None and i < len(captions):
                sample["caption"] = captions[i]
            else:
                sample["caption"] = ""
                
            data.append(sample)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return Dataset.from_list(data)

def create_test_batch(
    model,
    tokenizer,
    images: List[Union[str, Image.Image]],
    instruction: str
) -> Dict[str, torch.Tensor]:
    """
    Create a batch of test inputs for a vision-language model
    
    Args:
        model: Model to create inputs for
        tokenizer: Tokenizer for the model
        images: List of images (paths or PIL images)
        instruction: Text instruction
        
    Returns:
        Dictionary of model inputs
    """
    # Process images if they are paths
    processed_images = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        processed_images.append(img)
    
    # Create messages
    batch_inputs = []
    for img in processed_images:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        inputs = tokenizer(
            img,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        batch_inputs.append(inputs)
    
    # Batch inputs
    batched = {}
    for key in batch_inputs[0].keys():
        batched[key] = torch.cat([inputs[key] for inputs in batch_inputs], dim=0)
    
    # Move to device
    device = model.device
    for key in batched:
        batched[key] = batched[key].to(device)
    
    return batched