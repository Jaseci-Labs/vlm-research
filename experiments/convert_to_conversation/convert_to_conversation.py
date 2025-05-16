import os
import json
from typing import Optional, Dict, Union, Tuple
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image

def detect_split(dataset):
    if isinstance(dataset, DatasetDict):
        splits = {k: v for k, v in dataset.items() if k in ['train', 'validation', 'test']}
        print(f"Found splits: {list(splits.keys())}")
        return True, splits
    return False, {'dataset': dataset}

def split_data_function(dataset, train_percent=100.0, val_percent=0.0, test_percent=0.0, seed=42): 
    total_percent = train_percent + val_percent + test_percent
    if not (0 <= train_percent <= 100 and 0 <= val_percent <= 100 and 0 <= test_percent <= 100):
        raise ValueError("Percentages must be between 0 and 100.")
    if total_percent == 0:
        raise ValueError("At least one split percentage must be greater than 0.")
    
    if total_percent != 100:
        train_percent = (train_percent / total_percent) * 100
        val_percent = (val_percent / total_percent) * 100
        test_percent = (test_percent / total_percent) * 100
        print(f"Normalized percentages: train={train_percent:.1f}%, val={val_percent:.1f}%, test={test_percent:.1f}%")

    splits = {}
    remaining_data = dataset
    total_size = len(dataset)
    train_size = int((train_percent / 100) * total_size)
    val_size = int((val_percent / 100) * total_size)
    test_size = int((test_percent / 100) * total_size)

    print(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")

    if train_percent > 0:
        if val_percent > 0 or test_percent > 0:
            train, remaining_data = train_test_split(
                remaining_data, train_size=train_size, random_state=seed
            )
        else:
            train = remaining_data
        splits['train'] = train

    if val_percent > 0 and remaining_data:
        if test_percent > 0:
            val, remaining_data = train_test_split(
                remaining_data, train_size=val_size, random_state=seed
            )
        else:
            val = remaining_data
        splits['validation'] = val

    if test_percent > 0 and remaining_data:
        splits['test'] = remaining_data

    return splits

def make_serializable(obj):
    if isinstance(obj, Image.Image):
        return f"PIL.Image.Image(mode={obj.mode}, size={obj.size})"
    return str(obj) 

def format_as_unsloth(sample, split="train", caption_col=None, data_category="caption", image_col=None, image_path_prefix=None, prompt_text=None):
    if prompt_text is None:
            prompt_text = (
                "Answer the question based on the image." if data_category == "vqa"
                else "Describe this image in detail."
            )
    
    messages = []

    if split in ["train", "validation"]:

        user_content = []
        if image_col and image_col in sample and sample[image_col]:
            if image_path_prefix and isinstance(sample[image_col], str):
                image_path = os.path.join(image_path_prefix, sample[image_col])
                user_content.append({"type": "image_url", "image_url": {"url": image_path}})
            else:
                user_content.append({"type": "image", "image": sample[image_col]})
        
        user_content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": user_content})

        if caption_col and caption_col in sample and sample[caption_col]:
            messages.append({"role": "assistant", "content": sample[caption_col]})

    else:
        user_content = []
        if image_col and image_col in sample and sample[image_col]:
            if image_path_prefix and isinstance(sample[image_col], str):
                image_path = os.path.join(image_path_prefix, sample[image_col])
                user_content.append({"type": "image_url", "image_url": {"url": image_path}})
            else:
                user_content.append({"type": "image", "image": sample[image_col]})
        
        user_content.append({"type": "text", "text": prompt_text})
        messages.append({"role": "user", "content": user_content})

    return {"messages": messages}

def convert_to_unsloth(
    input_path_or_name,
    dataset_type="hf",  
    data_category="caption",  
    train_percent=100.0,
    val_percent=0.0,
    test_percent=0.0,
    prompt_text=None,
    caption_col=None,
    image_col=None,
    image_path_prefix=None
):  
    if image_col is None:
        image_col = "image"
    if caption_col is None:
        caption_col = "answers" if data_category == "vqa" else "caption"

    if dataset_type == "hf":
        try:
            dataset = load_dataset(input_path_or_name)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None   
    else:
        try:
            with open(input_path_or_name, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith('['): 
                    lines = json.loads(content)
                else: 
                    lines = [json.loads(l.strip()) for l in content.splitlines() if l.strip()]
            dataset = lines
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None    
    
    is_split, splits = detect_split(dataset)

    if is_split:
        available_splits = set(splits.keys())
        requested_splits = {
            name for name, pct in zip(["train", "validation", "test"], [train_percent, val_percent, test_percent]) if pct > 0
        }

        if not requested_splits.issubset(available_splits):
            combined_data = []
            for split_part in splits.values():
                combined_data.extend([dict(row) for row in split_part])
            splits = split_data_function(
                combined_data,
                train_percent=train_percent,
                val_percent=val_percent,
                test_percent=test_percent,
            )
            splits = {k: Dataset.from_list(v) for k, v in splits.items()}
    else:
        if isinstance(dataset, DatasetDict):
            dataset = [dict(row) for row in dataset]
        splits = split_data_function(
            dataset,
            train_percent=train_percent,
            val_percent=val_percent,
            test_percent=test_percent,
        )
        splits = {k: Dataset.from_list(v) for k, v in splits.items()}

    for split_name, split_data in splits.items():
        print(f"{split_name.capitalize()}: {len(split_data)} samples")

    formatted_outputs = {}

    for split_name, split_data in splits.items():
        if len(split_data) > 0:
            formatted_sample = format_as_unsloth(
                split_data[0],
                split=split_name,
                data_category=data_category,
                prompt_text=prompt_text,
                caption_col=caption_col,
                image_col=image_col,
                image_path_prefix=image_path_prefix
            )
            serializable_formatted = json.loads(json.dumps(formatted_sample, default=make_serializable))
            print(f"\nFormatted sample from '{split_name}' split:")
            print(json.dumps(serializable_formatted, indent=2, ensure_ascii=False))
            formatted_outputs[split_name] = formatted_sample

    return formatted_outputs

if __name__ == "__main__":
    formatted_outputs = convert_to_unsloth(
        input_path_or_name="unsloth/Radiology_mini",
        dataset_type="hf",
        data_category="caption",
        train_percent=60.0,
        val_percent=20.0,
        test_percent=20.0,
    )

