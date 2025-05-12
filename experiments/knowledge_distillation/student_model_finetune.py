from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported
from trl import SFTTrainer, SFTConfig

from PIL import Image
import json
import os

MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
JSON_FILE_PATH = "res.json"
IMAGES_PATH = "car_damage_dataset"
instruction = """
              Please provide response based on the provided output format. 
              Expected output format:\n```json\n{\n  \"predictions\": [\n    {\n      \"location\": \"front bumper\",\n      \"damage_type\": \"dent\",\n      \"severity\": \"major\"\n    },\n    {\n      \"location\": \"driver side door\",\n      \"damage_type\": \"scratch\",\n      \"severity\": \"minor\"\n    }\n  ],\n \"report\":\"Insurance Report: The vehicle sustained significant damage, including a major dent on the front bumper and a minor scratch on the driver side door. Estimated repair cost: $1,500.\"}\n```
              """


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


def configuration_for_training(model, tokenizer):
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = converted_dataset,
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
            report_to = "tensorboard",

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


# Main execution
if __name__ == "__main__":
    converted_dataset = get_custom_dataset(JSON_FILE_PATH)
    model, tokenizer = configure_model(MODEL_NAME)
    trainer = configuration_for_training(model, tokenizer)
    trainer.train()
    save_finetuned_model(model, tokenizer, "finetuned_model")