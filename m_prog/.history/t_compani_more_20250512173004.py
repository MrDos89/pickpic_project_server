from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()

IMAGE_FILENAMES = ["Golden_", "jar2.png"]
embedding1 = get_clip_embedding(IMAGE_FILENAMES[0])
embedding2 = get_clip_embedding(IMAGE_FILENAMES[1])

cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
print(f"두 이미지의 유사도: {cosine_similarity:.4f}")

if cosine_similarity >= 0.8:
    print("일치합니다.")
else:
    print("일치하지 않습니다.")