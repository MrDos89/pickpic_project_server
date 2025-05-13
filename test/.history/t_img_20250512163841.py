# import cv2
# from google.colab.patches import cv2_imshow
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
#   cv2_imshow(img)


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)

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

    # 이미지를 분류합니다.
    classification_result = classifier.classify(image)

    # 분류 결과를 처리합니다. (여기서는 예측 결과를 시각화합니다)
    images.append(image)
    top_category = classification_result.classifications[0].categories[0]
    predictions.append(f"{top_category.category_name} ({top_category.score:.2f})")

display_batch_of_images(images, predictions)