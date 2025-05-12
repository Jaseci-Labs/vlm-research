from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer

def load_model():
    task_path = "outputs/checkpoint-30" # put the output model path here

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
    image_path = "path_to_image.jpg"
    image = Image.open(image_path).convert("RGB")
    
    question = """
            Please provide response based on the provided output format. 
            Expected output format:\n```json\n{\n  \"predictions\": [\n    {\n      \"location\": \"front bumper\",\n      \"damage_type\": \"dent\",\n      \"severity\": \"major\"\n    },\n    {\n      \"location\": \"driver side door\",\n      \"damage_type\": \"scratch\",\n      \"severity\": \"minor\"\n    }\n  ],\n \"report\":\"Insurance Report: The vehicle sustained significant damage, including a major dent on the front bumper and a minor scratch on the driver side door. Estimated repair cost: $1,500.\"}\n```
            """
    
    model, tokenizer = load_model()    
    result = process_vqa(model, tokenizer, image, question)
    print(result)