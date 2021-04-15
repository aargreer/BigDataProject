import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_classifier_parameters():
    for i in range(1, 6):
        accuracy_avg = 0
        clf = RandomForestClassifier(max_depth=11, n_estimators=20, max_features=i)
        for j in range(0, 75):
            clf.fit(X_train, y_train.astype('int'))
            accuracy = clf.score(X_test, y_test)
            accuracy_avg += accuracy
            # print("Accuracy of decision tree classifier:", accuracy * 100, "%")
        print("Average accuracy for n=", i, "features:\t", accuracy_avg * 100 / 75, "%")


def output_to_pickle(output_clf):
    pickle.dump(output_clf, open("data/model.pickle", 'wb'))


def load_from_pickle():
    clf2 = pickle.load(open("data/model.pickle", 'rb'))
    return clf2


def predict_from_pickle(pickled_clf, test_instances):
    pickled_clf.predict(test_instances[:, 0:5])


clf = RandomForestClassifier(max_depth=11, n_estimators=20, max_features=3)

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

# X is all features besides the class variable, y is our classifier variable BURNCLAS
X, y = ds[:, 0:5], ds[:, 5]

# preprocess dataset, split into training and test part
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

# trains the model and outputs accuracy against test data
clf.fit(X_train, y_train.astype('int'))
accuracy = clf.score(X_test, y_test)
print("Accuracy:\t", accuracy * 100, "%")

# output trained model
output_to_pickle(clf)
