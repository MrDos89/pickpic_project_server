import os
import torch
from PIL import Image
import base64
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
from fastapi import Query

# 프로젝트 루트 기준 경로
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")

# 유사도 임계값 (기본값)
SIMILARITY_THRESHOLD = 0.25

def find_similar_images_by_clip(text: str, image_dir: str, features_dir: str, similarity_threshold: float = 0.0, detail: bool = False) -> Optional[List[dict]]:
    """
    띄어쓰기로 구분된 여러 키워드가 들어오면 각 키워드별로 영어로 번역 후 따로 검색해서
    모든 키워드에 해당하는 이미지를 반환 (중복 제거, 유사도는 최대값)
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')
    try:
        from googletrans import Translator
    except ImportError:
        raise ImportError('googletrans 패키지가 필요합니다. (pip install googletrans==4.0.0-rc1)')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # 이미지 임베딩(.npy) 파일 목록
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    if not feature_files:
        return None

    # npy 임베딩을 한 번에 numpy 배열로 로드 (벡터화)
    img_features = []
    img_filenames = []
    for f in feature_files:
        arr = np.load(os.path.join(features_dir, f))
        img_features.append(arr)
        img_filenames.append(f.replace('.npy', ''))
    img_features = np.stack(img_features, axis=0)  # shape: (N, 512)

    # 여러 키워드로 분리 및 번역
    keywords = text.strip().split()
    translator = Translator()
    translated_keywords = []
    for kw in keywords:
        try:
            translated = translator.translate(kw, src='ko', dest='en').text
            translated_keywords.append(translated)
        except Exception:
            translated_keywords.append(kw)  # 번역 실패시 원본 사용

    # 텍스트 임베딩도 모두 미리 구해서 벡터화
    text_embeds = []
    for keyword in translated_keywords:
        inputs = processor(text=[keyword], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs).cpu().numpy()[0]
        text_embeds.append(text_features)
    text_embeds = np.stack(text_embeds, axis=0)  # shape: (K, 512)

    # 유사도 계산 (브로드캐스팅)
    # img_features: (N, 512), text_embeds: (K, 512)
    img_norms = np.linalg.norm(img_features, axis=1, keepdims=True)  # (N, 1)
    text_norms = np.linalg.norm(text_embeds, axis=1, keepdims=True)  # (K, 1)
    # (K, N): 각 텍스트 임베딩과 모든 이미지 임베딩의 유사도
    sims = np.dot(text_embeds, img_features.T) / (text_norms * img_norms.T + 1e-8)

    # 이미지별로 {파일명: [(원본키워드, 번역키워드, 유사도), ...]} 저장
    image_keyword_scores = dict()
    for k_idx, keyword in enumerate(keywords):
        for n_idx, fname in enumerate(img_filenames):
            sim = sims[k_idx, n_idx]
            if sim < similarity_threshold:
                continue
            if fname not in image_keyword_scores:
                image_keyword_scores[fname] = []
            image_keyword_scores[fname].append((keyword, translated_keywords[k_idx], sim))

    # 정렬: 1) 매칭 키워드 개수 내림차순, 2) 유사도 합 내림차순
    sorted_images = sorted(
        image_keyword_scores.items(),
        key=lambda x: (len(x[1]), sum([s[2] for s in x[1]])),
        reverse=True
    )

    results = []
    for fname, matches in sorted_images:  
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
        if detail:
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            results.append({
                'filename': fname,
                'base64': img_base64,
                'matched_keywords': [m[0] for m in matches],
                'translated_keywords': [m[1] for m in matches],
                'scores': [float(m[2]) for m in matches],
                'matched_count': len(matches),
                'score_sum': float(sum([m[2] for m in matches]))
            })
        else:
            results.append({'filename': fname})
    return results if results else None

def save_clip_image_features(image_dir: str, features_dir: str, batch_size: int = 16):
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    os.makedirs(features_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for i in range(0, len(image_files), batch_size):
        batch_filenames = image_files[i:i+batch_size]
        batch_images = []
        valid_fnames = []
        for fname in batch_filenames:
            feature_path = os.path.join(features_dir, fname + ".npy")
            if os.path.exists(feature_path):
                continue
            img_path = os.path.join(image_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                valid_fnames.append(fname)
            except Exception as e:
                print(f"이미지 열기 실패: {fname} ({e})")
                continue
        if not batch_images:
            continue
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs).cpu().numpy()
        for fname, feat in zip(valid_fnames, image_features):
            feature_path = os.path.join(features_dir, fname + ".npy")
            np.save(feature_path, feat)
            print(f"저장 완료: {feature_path}")

class SearchQuery(BaseModel):
    text: str
    similarity_threshold: Optional[float] = 0.0