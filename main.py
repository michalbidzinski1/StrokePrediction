import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
train = pd.read_csv("healthcare-dataset-stroke-data.csv")
pd.set_option('display.max_columns', None)
print(train.head())
print("Liczba zduplikowanych wierszy:", train.duplicated().sum())

categorical = train.select_dtypes(include=['object']).columns.tolist()
for i in categorical:
    print(train[i].value_counts().to_frame(), '\n')

dataset = train.drop('id', axis=1)
dataset = dataset[dataset['gender'] != 'Other']
print("Table with the missing values of each feature: ", '\n', dataset.isna().sum())
dataset['bmi'].fillna(dataset['bmi'].mean(), inplace = True)
print("Table with the missing values of each feature: ", '\n', dataset.isna().sum())

# KNN
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
df = dataset[["age", "hypertension", "heart_disease","avg_glucose_level", "bmi", "stroke"]]
print(df)
predict = "stroke"
x = np.array(df.drop([predict], 1))
y = np.array(df[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
from sklearn import metrics
y_pred1 = knn.predict(x_test)
acc1 = metrics.accuracy_score(y_test, y_pred1)
accuracy_1_rounded = round(acc1*100, 2)
print("Accuracy k=1:", accuracy_1_rounded, "% \n")


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred1 = knn.predict(x_test)
acc3 = metrics.accuracy_score(y_test, y_pred1)
accuracy_3_rounded = round(acc3*100, 2)
print("Accuracy k=3:", accuracy_3_rounded, "% \n")
plot_confusion_matrix(knn, x_test, y_test, display_labels=["Less chance", "More chance"], cmap=plt.cm.YlOrRd)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred1 = knn.predict(x_test)
acc5 = metrics.accuracy_score(y_test, y_pred1)
accuracy_5_rounded = round(acc5*100, 2)
print("Accuracy k=5:", accuracy_5_rounded, "% \n")
# plot_confusion_matrix(knn, x_test, y_test, display_labels=["Less chance", "More chance"], cmap=plt.cm.YlOrRd)



labels = ['k = 1', 'k = 3', 'k = 5']
accuracy = [accuracy_1_rounded, accuracy_3_rounded, accuracy_5_rounded]

# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_axes([0,0,1,1])

# x = np.arange(len(accuracy))
# rects = ax.bar(x, accuracy)
#
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects)

# ax.bar(labels,accuracy)
# plt.ylabel('%', fontweight ='bold', fontsize = 15)
# plt.title("k najbliższych sąsiadów", fontsize=12)






# Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
accuracy_bayes = metrics.accuracy_score(y_test, y_pred)
accuracy_bayes_round = round(accuracy_bayes*100, 2)
# print("Accuracy Naive Bayes:", accuracy_bayes_round, "% \n")
# plot_confusion_matrix(gnb, x_test, y_test, display_labels=["Less chance", "More chance"], cmap=plt.cm.YlOrRd)


# Drzewa decyzyjne
from sklearn import metrics, tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_tree = accuracy_score(y_test, y_pred)
accuracy_tree_round = round(accuracy_tree*100,2)
print("Accuracy decision tree: ", accuracy_tree_round, "% \n")
print("Confusion matrix:")
# plot_confusion_matrix(clf, x_test, y_test, display_labels=["Less chance", "More chance"], cmap=plt.cm.YlOrRd)
# plt.show()
#
# tree.plot_tree(clf)
# plt.savefig('tree.pdf')


#Logistic Regression

from sklearn.linear_model import LogisticRegression
LogisticRegressionclf = LogisticRegression(random_state=0, max_iter = 400)
lr = LogisticRegressionclf.fit(x_train,y_train)
y_predict_test = LogisticRegressionclf.predict(x_test)

cm = confusion_matrix(y_test, y_predict_test)
accuracy_lg = accuracy_score(y_test, y_predict_test)
accuracy_lg = round(accuracy_lg*100,2)
print(classification_report(y_test, y_predict_test))
print('Accuracy Logistic Regression: ' , accuracy_score(y_test,y_predict_test))
# plot_confusion_matrix(lr, x_test, y_test, display_labels=["Less chance", "More chance"], cmap=plt.cm.YlOrRd)
# plt.show()

# labels = ['k najbliższych sąsiadów', 'Naive Bayes', 'Drzewa decyzyjne', "Regresja logistyczna", ]
# accuracy = [accuracy_3_rounded, accuracy_bayes_round, accuracy_tree_round, accuracy_lg]
#
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_axes([0,0,1,1])
#
# x = np.arange(len(accuracy))
# rects = ax.bar(x, accuracy)
#
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# autolabel(rects)
#
#
# plt.ylabel('%', fontweight ='bold', fontsize = 12)
# plt.title("Skuteczność klasyfikatorów", fontsize=12)
data_balance_check_labels = ['stroke = 0', 'stroke = 1']
total_instances_per_value = df['stroke'].value_counts()
pie_chart_colors = ['orange', 'red']
plt.figure(figsize=(6,6))
plt.pie(total_instances_per_value, labels = data_balance_check_labels, shadow = 1, explode = (0.1, 0), autopct='%1.2f%%', colors = pie_chart_colors)
# plt.show()
# plt.show()

#from tensorflow_addons import losses
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

# Training and testing neural network on the oversampled training data
nn_model = keras.Sequential([
    keras.layers.Dense(17, input_dim=17, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False, name='Adam'
), loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(x_train, y_train.values.ravel(), epochs=500)

print(nn_model.evaluate(x_test, y_test))

y_predicted_nn_model = nn_model.predict(x_test)
y_predicted_nn_model = np.round(y_predicted_nn_model)

print("Classification Report: \n", classification_report(y_test, y_predicted_nn_model))