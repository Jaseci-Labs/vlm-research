import torch
import torch.nn as nn
from omegaconf import DictConfig
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import io
import os

class DummyCOCODataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        # Synthetic data: dummy images and captions
        self.images = [Image.new('RGB', (64, 64), color='gray') for _ in range(size)]
        self.captions = [f"Object {i}" for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {'image': self.images[idx], 'text': self.captions[idx]}

def custom_collate_fn(batch):
    # Collate images (PIL.Image.Image) and texts (strings) into lists
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    return {'image': images, 'text': texts}

class VLM(nn.Module):
    def __init__(self, language_model, vision_model):
        super().__init__()
        self.language_model = language_model
        self.vision_model = vision_model
        self.projection = nn.Linear(768, language_model.config.hidden_size, dtype=torch.float16)  # CLIP to Qwen, match float16
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # Encode image
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_hidden_states = vision_outputs.pooler_output
        image_embeds = self.projection(image_hidden_states)
        
        # Combine with text (simplified: pass text only to language model)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        outputs = self.language_model(**inputs, labels=labels)
        return outputs

class VLMFinetuneExperiment:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")  # Force CPU for Intel Iris
        
        # Load models
        self.language_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model.pretrained_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model.pretrained_path)
        self.vision_model = CLIPVisionModel.from_pretrained(
            cfg.model.model.vision_model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(cfg.model.model.vision_model_path)
        
        # Combine into VLM
        self.model = VLM(self.language_model, self.vision_model)
        
        # Apply LoRA to language model only
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        self.model.to(self.device)
        
        # Dataset and loader
        self.dataset = DummyCOCODataset(size=cfg.dataset.dataset.size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.training.lr)
    
    def train(self):
        self.model.train()
        for epoch in range(1, self.cfg.training.epochs + 1):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in self.dataloader:
                images = batch['image']
                texts = batch['text']
                
                # Process inputs
                text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    pixel_values=image_inputs.pixel_values,
                    labels=text_inputs.input_ids
                )
                loss = outputs.loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Simplified accuracy
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == text_inputs.input_ids).float().mean().item()
                total += 1
            
            avg_loss = total_loss / len(self.dataloader)
            accuracy = correct / total if total > 0 else 0
            wandb.log({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save checkpoint
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.model.language_model.save_pretrained(f"{self.cfg.output_dir}/epoch_{epoch}")