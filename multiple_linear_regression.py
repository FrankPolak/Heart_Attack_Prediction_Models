# 1. Import Libraries and Dataset
import numpy as np
import pandas as pd

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2. Split Data into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 3. Train the Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)  # predictor
comparison = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

# 4. Evaluate the Model
correct = 0
for pred, test in comparison:
    if pred < 0.5 and test == 0 or pred >= 0.5 and test == 1:
        correct += 1

accuracy = correct / len(comparison)
print(accuracy) # 0.8217821782178217

# 5. Increase the Accuracy
correct = 0
inconclusive = 0
total = len(comparison)
for pred, test in comparison:
    if pred > 0.2 and pred < 0.8:
        inconclusive += 1
    elif pred < 0.5 and test == 0 or pred >= 0.5 and test == 1:
        correct += 1

accuracy = (correct + undecided) / total
print(accuracy)    # 0.8217821782178217
print(f"total: {total}")    # 101
print(f"correct: {correct}")    # 36
print(f"inconclusive: {inconclusive}")    # 62
print(f"wrong: {total - correct - inconclusive}")    # 3
