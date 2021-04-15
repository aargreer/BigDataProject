import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


clf = RandomForestClassifier(max_depth=7, n_estimators=12, max_features=5)

data = pd.read_csv("data/fire_data_2015.csv")

# uses a dict to convert from tree genus i.e. "Pinu", "Pice",... to 0, 1,...
counter = 0
tree_count_dict = {}
for i in data.iterrows():
    try:
        tree_count_dict[i[1]["tree_genus"]]
    except KeyError:
        tree_count_dict[i[1]["tree_genus"]] = counter
        counter += 1

# replace tree genus with ID and clamp fire value to 0 (no fire) or 1 (fire)
data = data.copy().replace(to_replace=tree_count_dict)
data = data.copy().replace(to_replace=[1, 2, 3, 4], value=1)

print("\n\t\t\tORIGINAL DATA:\n", data)

ds = data.to_numpy()

# X is a vertical slice of a tuple of features, y is our classifier variable BURNCLAS
X, y = ds[:, 0:5], ds[:, 5]

# preprocess dataset, split into training and test part
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# getting min and max values to determine mesh resolution
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

clf.fit(X_train, y_train.astype('int'))

print(X_test.ravel().shape, y_test.ravel().shape)
accuracy = clf.score(X_test, y_test)

print("Accuracy of decision tree classifier:", accuracy * 100, "%")
