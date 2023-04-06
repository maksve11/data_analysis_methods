import cv2
import numpy as np
import os
from imutils import paths
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


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

# # разбиваем данные на тренировочную и тестовую выборки в соотношении 75/25
# (trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=9)


# обучаем модель LinearSVC
model = LinearSVC(C=0.51, random_state=9)

# # обучение модели для 1-4 заданий
# model.fit(trainData, trainLabels)

# для последнего заданий
model.fit(data, labels)

# для заданий 1-4
# # оцениваем качество классификации на тестовой выборке
# acc = model.score(testData, testLabels)
# print("Accuracy: {:.2f}%".format(acc * 100))
#
# theta_257 = model.coef_[0][256]
# print("Значение коэффициента Thete_257 построенной гиперплоскости{:.5f}%", theta_257)
#
# theta_371 = model.coef_[0][370]
# print("Значение коэффициента Thete_371 построенной гиперплоскости{:.5f}%", theta_371)
#
# theta_125 = model.coef_[0][124]
# print("Значение коэффициента Thete_125 построенной гиперплоскости{:.5f}%", theta_125)
#
# # получаем предсказания модели на тестовых данных
# preds = model.predict(testData)
#
# # вычисляем матрицу ошибок
# cm = confusion_matrix(testLabels, preds)
# print("Матрица ошибок:\n", cm)
#
# # вычисляем точность, полноту и метрику F1 для каждого класса
# report = classification_report(testLabels, preds)
# print("Отчет по классификации:\n", report)
#
# # вычисляем среднее значение метрики F1 (Macro-F1)
# f1_scores = classification_report(testLabels, preds, output_dict=True)['macro avg']['f1-score']
# print("Среднее значение метрики F1 (Macro-F1): {:.2f}%".format(f1_scores*100))


# для последних заданий
test_path = 'test'

# проходим по каждому изображению в папке test
for filename in os.listdir(test_path):
    # проверяем, что файл является изображением
    if filename.endswith('.jpg'):
        # задаем полный путь к изображению
        image_path = os.path.join(test_path, filename)

        # загружаем изображение и изменяем его размер до 64x64 пикселей
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))

        # извлекаем гистограмму изображения и преобразуем ее в тензор
        hist = extract_histogram(image)
        image_tensor = np.array(hist).reshape((1, -1))

        # используем обученную модель для предсказания класса изображения
        prediction = model.predict(image_tensor)

        # выводим назначенный класс (0 для котов, 1 для собак)
        if prediction[0] == 0:
            print("Изображение", filename, "назначенный класс: cat - 0")
        else:
            print("Изображение", filename, "назначенный класс: dog - 1")