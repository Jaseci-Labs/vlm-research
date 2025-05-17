from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer
from huggingface_hub import login

def load_model():
    task_path = "Warun/Jaseci-Gemma-3-4B-Unsloth" # put the output model path here

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=task_path,
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def process_vqa(model, tokenizer, image, question):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    login(token="hf_XXXXXXXXXXXXXXXX")

    image_path = "path_of_your_image.jpg"  # replace with your image path
    image = Image.open(image_path).convert("RGB")
    
    question = "Descibe the damages of the car."
    
    model, tokenizer = load_model()    
    result = process_vqa(model, tokenizer, image, question)
    print(result)