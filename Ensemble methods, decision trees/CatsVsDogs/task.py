import os

import cv2
import numpy as np
from imutils import paths
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold


# Функция для извлечения гистограммы изображения
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# задаем путь к папке с изображениями
data_path = 'train'

# получаем список путей к изображениям и сортируем их по алфавиту
imagePaths = sorted(list(paths.list_images(data_path)))

# создаем список гистограмм и меток классов
data = []
labels = []

# проходим по каждому изображению в списке imagePaths
for imagePath in imagePaths:
    # загружаем изображение и изменяем его размер до 64x64 пикселей
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))

    # извлекаем гистограмму и добавляем ее в список данных
    hist = extract_histogram(image)
    data.append(hist)

    # получаем метку класса (0 - для кошек, 1 - для собак)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    if label == 'cat':
        labels.append(0)
    else:
        labels.append(1)

# преобразуем список данных и меток классов в массивы numpy
data = np.array(data)
labels = np.array(labels)

# Создаем базовые алгоритмы
svc = LinearSVC(C=1.74, random_state=220)
dtc = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_leaf_nodes=20, random_state=220)
rfc = RandomForestClassifier(n_estimators=19, criterion='entropy', min_samples_leaf=10, max_leaf_nodes=20, random_state=220)
bag = BaggingClassifier(base_estimator=dtc, n_estimators=19, random_state=220)

# Обучаем базовые алгоритмы
svc_scores = cross_val_score(svc, data, labels, cv=2)
print(f'SVM accuracy: {svc_scores.mean():.4f}')

bag_scores = cross_val_score(bag, data, labels, cv=2)
print(f'Bagging accuracy: {bag_scores.mean():.4f}')

rfc_scores = cross_val_score(rfc, data, labels, cv=2)
print(f'Random Forest accuracy: {rfc_scores.mean():.4f}')

# Обучаем метаалгоритм на базовых алгоритмах
meta_estimator = LogisticRegression(solver='lbfgs', random_state=220)
stack = StackingClassifier(
    estimators=[('svc', svc), ('bag', bag), ('rfc', rfc)],
    final_estimator=meta_estimator,
    cv=KFold(n_splits=2, shuffle=True, random_state=220)
)

stack.fit(data, labels)

stack_scores = cross_val_score(stack, data, labels, cv=2)
print(f'Stacking accuracy: {stack_scores.mean():.4f}')

image = cv2.imread('test/cat.1009.jpg')
image = cv2.resize(image, (64, 64))

# извлекаем гистограмму изображения
hist = extract_histogram(image)

# применяем метод predict_proba к гистограмме
# чтобы получить вероятности принадлежности изображения к классам 0 и 1
probabilities = stack.predict_proba([hist])[0]

# выводим вероятности
print(f'Вероятность принадлежности к классу 0: {probabilities[0]:.3f}')
print(f'Вероятность принадлежности к классу 1: {probabilities[1]:.3f}')
