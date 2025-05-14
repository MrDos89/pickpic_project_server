import os
import torch
from PIL import Image
import base64
import numpy as np
from typing import Optional, List

def find_similar_images_by_clip(text: str, image_dir: str, features_dir: str, top_n: int = 5, similarity_threshold: float = 0.0) -> Optional[List[dict]]:
    """
    띄어쓰기로 구분된 여러 키워드가 들어오면 각 키워드별로 영어로 번역 후 따로 검색해서
    모든 키워드에 해당하는 이미지를 top_n개씩 합쳐서 반환 (중복 제거, 유사도는 최대값)
    """
    try:
        from transformers import SiglipProcessor, SiglipModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')
    try:
        from googletrans import Translator
    except ImportError:
        raise ImportError('googletrans 패키지가 필요합니다. (pip install googletrans==4.0.0-rc1)')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiglipModel.from_pretrained('google/siglip-so400m-patch14-384').to(device)
    processor = SiglipProcessor.from_pretrained('google/siglip-so400m-patch14-384', use_fast=True)

    # 이미지 임베딩(.npy) 파일 목록
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    if not feature_files:
        return None

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

    # 이미지별로 {파일명: [(원본키워드, 번역키워드, 유사도), ...]} 저장
    image_keyword_scores = dict()

    for idx, keyword in enumerate(translated_keywords):
        inputs = processor(text=[keyword], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs).cpu().numpy()[0]

        for feat_file in feature_files:
            img_feature = np.load(os.path.join(features_dir, feat_file))
            sim = np.dot(text_features, img_feature) / (np.linalg.norm(text_features) * np.linalg.norm(img_feature))
            if sim < similarity_threshold:
                continue
            fname = feat_file.replace('.npy', '')
            if fname not in image_keyword_scores:
                image_keyword_scores[fname] = []
            image_keyword_scores[fname].append((keywords[idx], keyword, sim))

    # 정렬: 1) 매칭 키워드 개수 내림차순, 2) 유사도 합 내림차순
    sorted_images = sorted(
        image_keyword_scores.items(),
        key=lambda x: (len(x[1]), sum([s[2] for s in x[1]])),
        reverse=True
    )

    results = []
    for fname, matches in sorted_images[:top_n]:
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
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
    return results if results else None

def save_clip_image_features(image_dir: str, features_dir: str):
    try:
        from transformers import SiglipProcessor, SiglipModel
    except ImportError:
        raise ImportError('transformers, torch 패키지가 필요합니다.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiglipModel.from_pretrained('google/siglip-so400m-patch14-384').to(device)
    processor = SiglipProcessor.from_pretrained('google/siglip-so400m-patch14-384', use_fast=True)

    os.makedirs(features_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in image_files:
        feature_path = os.path.join(features_dir, fname + ".npy")
        if os.path.exists(feature_path):
            # 이미 임베딩이 존재하면 건너뜀
            # print(f"이미 존재: {feature_path} (건너뜀)")
            continue

        img_path = os.path.join(image_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 열기 실패: {fname} ({e})")
            continue

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs).cpu().numpy()[0]

        np.save(feature_path, image_features)
        print(f"저장 완료: {feature_path}")

def run_query(query):
    # 쿼리 실행 로직
    print(f"찾을 사진: {query}")
    # 실제 쿼리 실행 코드 작성

if __name__ == "__main__":
    query = input("찾을 사진: ")
    run_query(query)

    image_dir = "./backend/user_photos"
    features_dir = "./backend/features"
    top_n = 5
    similarity_threshold = 0.066

    results = find_similar_images_by_clip(query, image_dir, features_dir, top_n=top_n, similarity_threshold=similarity_threshold)
    if not results:
        print("유사한 이미지가 없습니다.")
    else:
        for r in results:
            # print(f"파일명: {r['filename']}, 매칭 키워드: {r['matched_keywords']}, 유사도: {r['scores']}")
            img_path = os.path.join(image_dir, r['filename'])
            img = Image.open(img_path)
            img.show()

    save_clip_image_features(image_dir, features_dir) 