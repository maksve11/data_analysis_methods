import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
import joblib

# Загрузка данных
data = pd.read_csv('candy-data.csv', delimiter=',')

# Удаление строк с указанными конфетами
data = data[(data['competitorname'] != 'Fun Dip') & (data['competitorname'] != 'Laffy Taffy') & (data['competitorname'] != 'Peanut butter M&Ms')]

# Выбор предикторов и отклика
X = data[['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent']]
y = data[['Y']]

lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

# Обучение модели логистической регрессии
model = LinearRegression().fit(X, y_transformed)

# сохранение модели
joblib.dump(model, 'model.pkl')

# Загрузка тестовых данных
test_data = pd.read_csv('candy-test.csv', delimiter=',')

# выполнение прогнозов
predictions = []
for index, row in test_data.iterrows():
    features = row[1:-1].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    predictions.append(prediction)
    print(row['competitorname'], prediction)

# оценка модели
y_true = test_data.iloc[:, -1].values
cm = confusion_matrix(y_true, predictions)
precision = precision_score(y_true, predictions)
recall = recall_score(y_true, predictions)
auc = roc_auc_score(y_true, model.predict_proba(test_data.iloc[:, 1:-1])[:, 1])

print('Confusion matrix:')
print(cm)
print('Precision:', precision)
print('Recall:', recall)
print('AUC:', auc)

#
# # Выбор предикторов тестовых данных
# X_test = test_data[['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent']]
#
# # Предсказание вероятности отнесения конфеты Trolli Sour Bites к классу 1
# trolli_index = test_data[test_data['competitorname'] == 'Sugar Daddy'].index[0]
# trolli_prob = model.predict_proba(X_test)[trolli_index][1]
# print(trolli_prob)