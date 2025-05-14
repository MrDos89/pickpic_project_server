import clip
import torch
from PIL import Image

# CLIP 모델 및 변환기 불러오기
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 로드 및 변환
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# 텍스트 입력
texts = ["a photo of a dog", "a photo of a cat"]
text_tokens = clip.tokenize(texts).to(device)

# 이미지 및 텍스트 특징 추출
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

# 유사도 계산
similarity = (image_features @ text_features.T).softmax(dim=-1)
print(similarity)