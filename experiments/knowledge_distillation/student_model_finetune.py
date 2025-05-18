from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported
from trl import SFTTrainer, SFTConfig
import torch

from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
import json
import os

from huggingface_hub import login
import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True
wandb.init(
    project="Gemma 3 4B Test Unsloth",
    name="Gemma test Unsloth",
    group="gemma tests",
    tags=["gemma","vision", "finetune"],
    notes="Testing gemma 3 4B Unsloth",
    config={
        "model": "Gemma 3 4B",
    },
)

MODEL_NAME = "unsloth/gemma-3-4b-it"
JSON_FILE_PATH = "res.json"
IMAGES_PATH = "image_dataset"
instruction = "Descibe the damages of the car.",


def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(sample["caption"], indent=2)}],
            },
        ]
    }

# should be edited
def get_custom_dataset(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    custom_dataset = []
    # here data is list of key value pairs
    for sample in data:
        full_path = sample
        if os.path.exists(full_path):
            try:
                image = Image.open(full_path)
                custom_dataset.append(
                    convert_to_conversation(
                        {"image": image, "caption": data[sample]}
                    )
                )
            except Exception as e:
                print(f"[ERROR] Could not load image {full_path}: {e}")
        else:
            print(f"[WARNING] Image not found: {full_path}")
    return custom_dataset


def configure_model(MODEL_NAME):
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit = False,
        use_gradient_checkpointing = "unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    return model, tokenizer


def configuration_for_training(model, tokenizer, train_data, val_data):
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = train_data,
        eval_dataset = val_data,
        args = SFTConfig(

            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 30,
            # num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb",

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
    )
    return trainer

def save_finetuned_model(model, tokenizer, model_name):
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)


def upload_to_huggingface_hub(model, processor):
    model.push_to_hub("Warun/Jaseci-Gemma-3-4B-Unsloth")
    processor.push_to_hub("Warun/Jaseci-Gemma-3-4B-Unsloth")


# Main execution
if __name__ == "__main__":
    login(token="hf_XXXXXXXXXXXXXXXX")
    full_dataset = get_custom_dataset(JSON_FILE_PATH)
    train_data, test_val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=42)

    model, tokenizer = configure_model(MODEL_NAME)
    trainer = configuration_for_training(model, tokenizer, train_data, val_data)

    # start memory
    print("Starting training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Peak reserved memory = {used_memory} GB.")

    save_finetuned_model(model, tokenizer, "finetuned_model")
    trainer.push_to_hub()
    upload_to_huggingface_hub(model, tokenizer)
