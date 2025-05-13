import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# 모델 경로
MODEL_PATH = 'model\mobilenet_v3_small.tflite'

# 임베더 옵션 설정
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageEmbedderOptions(base_options=base_options)
embedder = vision.ImageEmbedder.create_from_options(options)

def get_embedding(image_path):
    image = mp.Image.create_from_file(image_path)
    result = embedder.embed(image)
    return np.array(result.embeddings[0].embedding)

# 비교할 이미지 파일명
IMAGE_FILENAMES = ["jar.jp", "jar2.jpg"]

# 임베딩 추출
embedding1 = get_embedding(IMAGE_FILENAMES[0])
embedding2 = get_embedding(IMAGE_FILENAMES[1])

# 코사인 유사도 계산
cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
print(f"두 이미지의 유사도: {cosine_similarity:.4f}")