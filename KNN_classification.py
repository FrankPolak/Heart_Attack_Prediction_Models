# 1. Import Libraries and Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, [3,7]].values
y = dataset.iloc[:, -1].values

# 2. Split Data into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 3. Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. Train the KNN Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 18, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)th

y_pred = classifier.predict(X_test)  # predictor

# 5. Evaluate the Model
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(y_test, y_pred) # 0.7227722772277227

# 6. Visualise the Results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train   # sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 3, stop = X_set[:, 0].max() + 3, step = 1),
                     np.arange(start = X_set[:, 1].min() - 5, stop = X_set[:, 1].max() + 5, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('green', 'red'))(i), label = j)
plt.title('K-NN (Training set)')    # plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate')
plt.legend()
plt.show()
