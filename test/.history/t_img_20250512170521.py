
# STEP 1: 필요모듈 가져오기
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='model/efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

IMAGE_FILENAMES = ["dah.jpg", "cat_and_dog.jpg"]  # 실제 파일명으로 수정하세요

def display_batch_of_images(images, predictions):
    for i, pred in enumerate(predictions):
        print(f"Image {i}: {pred}")

images = []
predictions = []
for image_name in IMAGE_FILENAMES:
  # STEP 3: 입력 이미지 가져오기
  image = mp.Image.create_from_file(image_name)

  # STEP 4: 이미지 분류
  classification_result = classifier.classify(image)

  # STEP 5: 분류결과 처리
  images.append(image)
  top_category = classification_result.classifications[0].categories[0]
  predictions.append(f"{top_category.category_name} ({top_category.score:.2f})")

display_batch_of_images(images, predictions)