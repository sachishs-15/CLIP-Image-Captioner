import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": path/to/video},
            {"type": "text", "text": "What is happening in this video?"},
        ]
    }
]

inputs = processor.apply_chat_template([messages], add_generation_prompt=True)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)