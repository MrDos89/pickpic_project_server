import argparse
from PIL import Image
from .utils.image_search import find_similar_images_by_clip, save_clip_image_features
from .utils.config import DATA_DIR, TEMP_DIR, SIMILARITY_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description='이미지 검색 CLI')
    parser.add_argument('--query', '-q', type=str, help='검색할 키워드')
    parser.add_argument('--update', '-u', action='store_true', help='이미지 특징 벡터 업데이트')
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD, 
                       help=f'유사도 임계값 (기본값: {SIMILARITY_THRESHOLD})')
    
    args = parser.parse_args()

    if args.update:
        print("이미지 특징 벡터를 업데이트합니다...")
        save_clip_image_features(DATA_DIR, TEMP_DIR)
        print("업데이트가 완료되었습니다.")
        return

    if not args.query:
        parser.print_help()
        return

    print(f"검색어: {args.query}")
    results = find_similar_images_by_clip(
        args.query,
        DATA_DIR,
        TEMP_DIR,
        similarity_threshold=args.threshold
    )

    if not results:
        print("유사한 이미지를 찾을 수 없습니다.")
        return

    print(f"\n검색 결과 ({len(results)}개):")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['filename']}")
        print(f"   매칭된 키워드: {', '.join(r['matched_keywords'])}")
        print(f"   번역된 키워드: {', '.join(r['translated_keywords'])}")
        print(f"   유사도 점수: {r['score_sum']:.4f}")
        
        # 이미지 표시
        img_path = f"{DATA_DIR}/{r['filename']}"
        try:
            img = Image.open(img_path)
            img.show()
        except Exception as e:
            print(f"   이미지 표시 실패: {e}")

if __name__ == "__main__":
    main() 