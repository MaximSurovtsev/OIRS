import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from imutils import paths

MODEL = 'output/model.h5'
RESULT = 'data/result'
TEST_IMAGES = 'data/test'

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

model = load_model(MODEL)
test_images = list(paths.list_images(TEST_IMAGES))

for idx, test_image in enumerate(test_images):
    image = cv2.imread(test_image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    # image = np.expand_dims(image, axis=-1)
    image = image.astype("float") / 255.0
    image = np.array([image])

    preds = model.predict(image)
    original = (image * 255).astype("uint8")
    recon = (preds * 255).astype("uint8")

    # объединить тестовую картинку с восстановленной
    output = np.hstack([original[0], recon[0]])
    file_path = os.path.join(RESULT, f'{idx}.png')
    cv2.imwrite(file_path, output)
