import os
import torch
import gc
import unsloth
import pandas as pd
import shutil
from PIL import Image
import random
from transformers import TrainerCallback
from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import numpy as np
import time
import trl

# Debug: Print trl version
print(f"Using trl version: {trl.__version__}")

# ----------- Settings ----------- #
EXCEL_PATH = "hyperparameters.xlsx"
DATASET_SIZE = 1000   # Training dataset size (for quick test)
VAL_SIZE = 200       # Validation dataset size
TEST_SIZE = 200      # Test dataset size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_FINE_TUNE = True  # Force fine-tuning for testing
OUTPUT_DIR = "/workspace/outputs"
INFO_DIR = "/workspace/info"  # Directory for loss and prediction CSVs

# Ensure info directory exists
os.makedirs(INFO_DIR, exist_ok=True)

instruction = (
    "In one concise sentence, summarize key objects and layout of this aerial scene."
)

# ----------- Custom Callback for Loss Logging ----------- #
class LossLoggingCallback(TrainerCallback):
    def __init__(self, csv_path):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.csv_path = csv_path
        # Initialize CSV with headers
        pd.DataFrame(columns=["Step", "Training_Loss", "Validation_Loss"]).to_csv(self.csv_path, index=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            train_loss = logs["loss"]
            self.train_losses.append(train_loss)
            self.steps.append(step)
            print(f"Step {step}: Training Loss = {train_loss}")
            # Append to CSV
            df = pd.DataFrame({
                "Step": [step],
                "Training_Loss": [train_loss],
                "Validation_Loss": [self.eval_losses[-1] if self.eval_losses else float("nan")]
            })
            df.to_csv(self.csv_path, mode="a", header=False, index=False)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            step = state.global_step
            eval_loss = metrics["eval_loss"]
            self.eval_losses.append(eval_loss)
            print(f"Step {step}: Validation Loss = {eval_loss}")
            # Update the latest row in CSV with validation loss
            if self.steps and self.steps[-1] == step:
                df = pd.read_csv(self.csv_path)
                df.loc[df["Step"] == step, "Validation_Loss"] = eval_loss
                df.to_csv(self.csv_path, index=False)
            else:
                # Append new row if no training loss for this step
                df = pd.DataFrame({
                    "Step": [step],
                    "Training_Loss": [float("nan")],
                    "Validation_Loss": [eval_loss]
                })
                df.to_csv(self.csv_path, mode="a", header=False, index=False)

# ----------- Functions ----------- #

def convert_to_conversation(image, caption):
    """Convert an image and caption to conversation format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]
    }

def load_rsicd_dataset(split="valid", size_limit=None):
    """Load RSICD dataset from Hugging Face, selecting one random caption per image."""
    dataset = load_dataset("arampacha/rsicd", split=split)
    print(f"Loading RSICD dataset ({split} split): {len(dataset)} entries found")
    
    processed_data = []
    for entry in dataset:
        image = entry["image"]
        captions = entry["captions"]  # List of captions
        if not captions:  # Skip if no captions
            continue
        # Select one random caption
        selected_caption = random.choice(captions)
        image = image.resize((224, 224))  # Resize to 224x224
        # Create a single sample with the selected caption
        processed_data.append(convert_to_conversation(image, selected_caption))
    
    if size_limit:
        processed_data = processed_data[:size_limit]
    
    print(f"Processed dataset with {len(processed_data)} samples (one caption per image)")
    return processed_data

def load_rsicd_test_dataset(split="test"):
    """Load RSICD test dataset for evaluation, keeping all captions."""
    dataset = load_dataset("arampacha/rsicd", split=split)
    print(f"Loading RSICD test dataset: {len(dataset)} entries found")
    
    test_data = []
    for entry in dataset:
        image = entry["image"]
        captions = entry["captions"]  # List of all captions (typically 5)
        image = image.resize((224, 224))
        test_data.append({"image": image, "captions": captions})  # Store all captions
    
    # Limit test dataset to TEST_SIZE
    test_data = test_data[:TEST_SIZE]
    
    print(f"Test dataset loaded with {len(test_data)} entries")
    return test_data

def evaluate_model(sentence_model, model, tokenizer, test_dataset, params):
    """Evaluate the model on the test dataset using max cosine similarity across all captions and save predictions."""
    results = []
    generated_texts = []
    all_ground_truths = []
    max_similarities = []

    for idx, sample in enumerate(test_dataset):
        image = sample["image"]
        gt_captions = sample["captions"]

        # Prepare input for model
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            [image],
            [input_text],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(DEVICE)

        # Generate caption
        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        output = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        if "assistant" in generated:
            generated = generated.split("assistant")[-1].strip()
        generated_texts.append(generated)
        all_ground_truths.append(gt_captions)

        # Compute embeddings for generated caption
        gen_embedding = sentence_model.encode([generated], convert_to_tensor=True)

        # Compute embeddings for all ground-truth captions
        gt_embeddings = sentence_model.encode(gt_captions, convert_to_tensor=True)

        # Calculate cosine similarities
        similarities = util.cos_sim(gen_embedding, gt_embeddings)[0]
        max_similarity = similarities.max().item()
        max_similarities.append(max_similarity)

        # Store results
        results.append({
            "image_index": idx,
            "generated_caption": generated,
            "ground_truth_captions": "; ".join(gt_captions),
            "max_similarity": max_similarity
        })

    # Calculate mean of maximum similarities
    mean_score = np.mean(max_similarities)
    print(f"Generated texts: {len(generated_texts)}")
    print(f"Ground truth caption sets: {len(all_ground_truths)}")
    print(f"Maximum similarities: {len(max_similarities)}")

    # Save results to CSV
    lr_str = f"{params['learning_rate']}".replace(".", "p")
    csv_name = f"predictions_lr{lr_str}_bs{params['batch_size']}_epochs{params['num_train_epochs']}.csv"
    csv_path = os.path.join(INFO_DIR, csv_name)
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    return round(mean_score, 4)

def fine_tune_model(params: dict):
    """Fine-tune the model with validation every step, saving all losses to CSV."""
    print(
        f"Training model: {params['model_name']} with LR={params['learning_rate']}, Epochs={params['num_train_epochs']}"
    )

    # Generate CSV file name for losses: learning-rate-batch-size-epoch.csv
    lr_str = f"{params['learning_rate']}".replace(".", "p")
    csv_name = f"losses_lr{lr_str}_bs{params['batch_size']}_epochs{params['num_train_epochs']}.csv"
    csv_path = os.path.join(INFO_DIR, csv_name)
    print(f"Saving loss logs to: {csv_path}")

    # Debug: Check cache locations
    hf_cache = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    print(f"Hugging Face cache: {hf_cache}")
    if os.path.exists(hf_cache):
        print(f"Hugging Face cache contents: {os.listdir(hf_cache)}")
    unsloth_cache = "/workspace/unsloth_compiled_cache"
    print(f"Unsloth cache: {unsloth_cache}")
    if os.path.exists(unsloth_cache):
        print(f"Unsloth cache contents: {os.listdir(unsloth_cache)}")

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Record start time
    start_time = time.time()

    # Load the model and tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        params["model_name"],
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # Define directories
    cache_dir = os.path.join(hf_cache, f"models--{params['model_name'].replace('/', '--')}")
    output_dir = os.path.abspath(OUTPUT_DIR)
    unsloth_cache = "/workspace/unsloth_compiled_cache"

    # Debug: Check directories
    print(f"\nCache directory: {cache_dir}")
    print(f"Cache exists before: {os.path.exists(cache_dir)}")
    if os.path.exists(cache_dir):
        print(f"Cache contents: {os.listdir(cache_dir)}")
    print(f"Output directory: {output_dir}")
    print(f"Output exists before: {os.path.exists(output_dir)}")
    if os.path.exists(output_dir):
        print(f"Output contents: {os.listdir(output_dir)}")
    print(f"Unsloth cache directory: {unsloth_cache}")
    print(f"Unsloth cache exists before: {os.path.exists(unsloth_cache)}")
    if os.path.exists(unsloth_cache):
        print(f"Unsloth cache contents: {os.listdir(unsloth_cache)}")

    # Fine-tuning setup with LoRA parameters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        random_state=3407,
    )

    # Load datasets
    train_dataset = load_rsicd_dataset(split="train", size_limit=DATASET_SIZE)  # Training on train split
    valid_dataset = load_rsicd_dataset(split="valid", size_limit=VAL_SIZE)      # Validation on valid split
    test_dataset = load_rsicd_test_dataset(split="test")                        # Test on RSICD test split
    FastVisionModel.for_training(model)

    # Check if validation dataset is empty
    if not valid_dataset:
        print("Warning: Validation dataset is empty. Validation loss will not be computed.")
        last_valid_loss = 0.0
    else:
        print(f"Validation dataset loaded with {len(valid_dataset)} entries")

    # Initialize the loss logging callback with CSV path
    loss_callback = LossLoggingCallback(csv_path=csv_path)

    # Use SFTConfig with validation every step
    sft_config = SFTConfig(
        per_device_train_batch_size=int(params["batch_size"]),
        gradient_accumulation_steps=4,
        num_train_epochs=float(params["num_train_epochs"]),
        learning_rate=float(params["learning_rate"]),
        warmup_steps=int(params["warmup_steps"]),
        weight_decay=float(params["weight_decay"]),
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=2048,
        eval_strategy="steps",  
        eval_steps=1,          # Changed from 5 to 1 for per-step validation
        per_device_eval_batch_size=int(params["batch_size"]),
    )

    # Create trainer with validation dataset
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=sft_config,
        callbacks=[loss_callback],
    )

    # Run training for all epochs
    trainer.train()

    # Collect validation losses from callback
    eval_losses = loss_callback.eval_losses

    # Record end time and calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    training_time_minutes = training_time / 60

    # Get last losses
    last_train_loss = loss_callback.train_losses[-1] if loss_callback.train_losses else 0.0
    last_valid_loss = loss_callback.eval_losses[-1] if loss_callback.eval_losses else 0.0
    print(f"Last Training Loss: {last_train_loss:.4f}")
    print(f"Last Validation Loss: {last_valid_loss:.4f}")

    # Measure peak memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak Memory Usage: {peak_memory:.2f} GB")
    else:
        peak_memory = 0.0
        print("No GPU available, memory usage not tracked.")

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    accuracy = evaluate_model(sentence_model, model, tokenizer, test_dataset, params)

    # Clean up memory
    print("\nCleaning up memory...")
    del model
    del tokenizer
    del sentence_model

    # Clear references to datasets
    train_dataset = None
    valid_dataset = None
    test_dataset = None
    
    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory stats

    # Delete cache and output directories (excluding info)
    for dir_path in [cache_dir, output_dir, unsloth_cache]:
        if dir_path != INFO_DIR:
            print(f"\nAttempting to delete {dir_path}...")
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path, ignore_errors=False)
                    print(f"Successfully deleted {dir_path}")
                except Exception as e:
                    print(f"Failed to delete {dir_path}: {str(e)}")
                    print(f"Contents: {os.listdir(dir_path) if os.path.exists(dir_path) else 'N/A'}")
            else:
                print(f"Directory does not exist: {dir_path}")

    return accuracy, last_train_loss, last_valid_loss, training_time_minutes, peak_memory

def main():
    """Main function to run hyperparameter tuning."""
    # Read hyperparameters from Excel
    try:
        df = pd.read_excel(EXCEL_PATH)
        print("Loaded DataFrame:")
        print(df)
    except FileNotFoundError:
        print(f"Error: {EXCEL_PATH} not found. Please provide the Excel file with hyperparameters.")
        return

    # Ensure required columns exist
    required_columns = ["model_name", "learning_rate", "batch_size", "num_train_epochs", "warmup_steps", "weight_decay"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in {EXCEL_PATH}")
            return

    # Ensure output columns exist with object dtype to handle mixed types
    for col in ["accuracy", "training_loss", "validation_loss", "training_time", "memory_usage"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    for idx, row in df.iterrows():
        if not FORCE_FINE_TUNE and pd.notna(row.get("accuracy")):
            print(f"Skipping row {idx}: already trained.")
            continue

        try:
            hyperparams = {
                "model_name": row["model_name"],
                "learning_rate": row["learning_rate"],
                "batch_size": row["batch_size"],
                "num_train_epochs": row["num_train_epochs"],
                "warmup_steps": row["warmup_steps"],
                "weight_decay": row["weight_decay"]
            }
            for key, value in hyperparams.items():
                if pd.isna(value):
                    raise ValueError(f"Missing or invalid value for '{key}' in row {idx}")

            acc, train_loss, valid_loss, train_time, mem_usage = fine_tune_model(hyperparams)
            df.at[idx, "accuracy"] = acc
            df.at[idx, "training_loss"] = train_loss
            df.at[idx, "validation_loss"] = valid_loss
            df.at[idx, "training_time"] = round(train_time, 2)
            df.at[idx, "memory_usage"] = round(mem_usage, 2)
            print(f"Row {idx} trained. Accuracy: {acc}, Training Loss: {train_loss:.4f}, "
                  f"Validation Loss: {valid_loss:.4f}, Training Time: {train_time:.2f} minutes, "
                  f"Memory Usage: {mem_usage:.2f} GB")

        except Exception as e:
            print(f"Error in row {idx}: {str(e)}")
            df.at[idx, "accuracy"] = f"Error: {str(e)}"
            df.at[idx, "training_loss"] = "Error"
            df.at[idx, "validation_loss"] = "Error"
            df.at[idx, "training_time"] = "Error"
            df.at[idx, "memory_usage"] = "Error"

        # Save to Excel after each run
        df.to_excel(EXCEL_PATH, index=False)
        print(f"Excel file updated: {EXCEL_PATH}")

    print("All fine-tuning completed!")

if __name__ == "__main__":
    main()