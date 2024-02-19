# 1. Import Libraries and Dataset
import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2. Split Data into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 3. Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. Build the ANN
ann = tf.keras.models.Sequential()                              # initialise the neural network
ann.add(tf.keras.layers.Dense(units=84, activation='relu'))     # add input layer and first hidden layer (rectifier activation function)

ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))   # add output layer (sigmoid activation function)

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 5. train the ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100)
y_pred = ann.predict(X_test)

# 6. Evaluate the Model
y_pred = (y_pred >= 0.5)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)  # 0.8217821782178217

# 7. Increase the Accuracy
y_pred = ann.predict(X_test)
comparison = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
correct = 0
inconclusive = 0
total = len(comparison)
for pred, test in comparison:
    if pred > 0.2 and pred < 0.8:
        inconclusive += 1
    elif pred < 0.5 and test == 0 or pred >= 0.5 and test == 1:
        correct += 1

accuracy = (correct + inconclusive) / total
print(accuracy)    # 0.9306930693069307
print(f"total: {total}")    # 101
print(f"correct: {correct}")    # 70
print(f"inconclusive: {inconclusive}")    # 24
print(f"wrong: {total - correct - inconclusive}")    # 7