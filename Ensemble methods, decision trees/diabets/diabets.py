import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

# Загрузка данных
data = pd.read_csv('diabetes.csv')
train_data = data[:620] # первые 620 строк для тренировки
test_data = data[620:] # остальные для тестирования

# Выбор предикторов и отклика
predictors = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = 'Outcome'

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(train_data[predictors], train_data[target], test_size=0.2, random_state=2020)

# Обучение модели
clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=15, min_samples_leaf=10, random_state=2020)
clf.fit(X_train, y_train)

# Оценка модели на тестовых данных
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

# Вывод результатов
print('Accuracy:', accuracy)
print('F1 score (Macro-F1):', f1)

# Визуализация дерева решений
columns = list(X_train.columns)
export_graphviz(clf, out_file='tree.dot',
                feature_names=columns,
                class_names=['0', '1'],
                rounded=True, proportion=False,
                precision=2, filled=True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)