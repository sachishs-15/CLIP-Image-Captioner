import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datasets import load_metric
import random
from tqdm import tqdm

# ------------------ PART B.1 ------------------ #
# ✅ Apply patch-wise occlusion
def occlude_image(image, mask_percentage, patch_size=16):
    """
    Args:
        image (PIL.Image or np.array): Input image
        mask_percentage (int): % of patches to occlude (e.g., 10, 50, 80)
        patch_size (int): Size of patch (default = 16)
    Returns:
        np.array: Occluded image as numpy array
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    h, w, c = image.shape
    h_patches = h // patch_size
    w_patches = w // patch_size

    total_patches = h_patches * w_patches
    num_to_mask = int((mask_percentage / 100.0) * total_patches)

    mask_indices = random.sample(range(total_patches), num_to_mask)

    occluded = image.copy()

    for idx in mask_indices:
        row = idx // w_patches
        col = idx % w_patches
        occluded[
            row * patch_size:(row + 1) * patch_size,
            col * patch_size:(col + 1) * patch_size,
            :
        ] = 0  # black mask

    return Image.fromarray(occluded)


# ------------------ PART B.2 ------------------ #
# ✅ Evaluate model on occluded images
def evaluate_on_occluded_images(model, dataloader, device, occlusion_levels=[10, 50, 80], tokenizer=None):
    """
    Args:
        model: Your trained image captioning model (custom or SmolVLM)
        dataloader: Test dataloader with image paths and ground truth captions
        device: 'cuda' or 'cpu'
        occlusion_levels: List of occlusion percentages to evaluate
        tokenizer: tokenizer/processor for decoding (optional if needed)
    Returns:
        Dictionary: For each occlusion level → metric delta
    """
    model.eval()
    model.to(device)

    base_scores = {'bleu': [], 'rouge': [], 'meteor': []}
    results = {}

    for occ in occlusion_levels:
        print(f"Evaluating for {occ}% occlusion...")
        predictions, references = [], []

        for batch in tqdm(dataloader):
            images, caps = batch['image'], batch['caption']
            for img, ref_caption in zip(images, caps):
                # Apply occlusion
                occluded_img = occlude_image(img, occ)

                # Apply transforms if needed (resize, normalize etc.)
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor()
                ])
                image_tensor = transform(occluded_img).unsqueeze(0).to(device)

                # Run inference
                with torch.no_grad():
                    output = model.generate(image_tensor)  # or use your model's interface
                    if tokenizer:
                        caption = tokenizer.decode(output[0], skip_special_tokens=True)
                    else:
                        caption = output[0]

                predictions.append(caption)
                references.append([ref_caption])

        # Compute metrics
        bleu = load_metric("bleu").compute(predictions=predictions, references=references)
        rouge = load_metric("rouge").compute(predictions=predictions, references=references)
        meteor = load_metric("meteor").compute(predictions=predictions, references=references)

        results[occ] = {
            "BLEU": bleu["bleu"],
            "ROUGE-L": rouge["rougeL"],
            "METEOR": meteor["meteor"]
        }

    return results
