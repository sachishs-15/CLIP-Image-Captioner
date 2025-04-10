import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_metric

def compute_metrics(predictions, references):

    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)

    return {
        "bleu": bleu_score,
        "rouge": rouge_score,
        "meteor": meteor_score,
    }

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/radahn/Sachish/try/DL-Ass-2/images/vgg16.png"},
            {"type": "text", "text": "Generate a caption for this image."},
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts[0])

# Report BLEU, ROUGE-L, and METEOR scores as a baseline.
