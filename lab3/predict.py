from keras.models import load_model
from imutils import paths
import pickle
import cv2
import os
import matplotlib.pyplot as plt

TEST_IMAGES = 'data/test'
RESULT = 'data/result'
MODEL = 'output/model.pickle'
LABEL = 'output/labels.pickle'

# загружаем модель и метки классов
print("[INFO] loading network and label...")
model = load_model(MODEL)
class_labels = pickle.loads(open(LABEL, "rb").read())

# цикл по изображениям
test_images = list(paths.list_images(TEST_IMAGES))

for idx, test_image in enumerate(test_images):
    # загружаем входное изображение и меняем его размер на необходимый
    image = cv2.imread(test_image)
    output = image.copy()

    image = cv2.resize(image, (32, 32)).flatten()
    # масштабируем значения пикселей к диапазону [0, 1]
    image = image.astype("float") / 255.0
    image = image.reshape((1, image.shape[0]))

    # массив верояностей, к какому классу относится изображение
    preds = model.predict(image)

    # находим индекс класса с наибольшей вероятностью и его название
    i = preds.argmax(axis=1)[0]
    class_label = class_labels.classes_[i]

    # класс + вероятность на выходном изображении (округление до второго знака)
    text = f'{class_label}: {round(preds[0][i] * 100, 2)}%'

    cv2.putText(output, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    plt.imsave(os.path.join(RESULT, f'{idx}.png'), cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    print(f'{test_image} is {class_label}')
