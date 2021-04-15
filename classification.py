# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB()]

X, y = make_classification(n_features=5, n_redundant=0, n_informative=5, random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

data = pd.read_csv("fire_data_2011.csv")

# uses a dict to convert from tree genus i.e. "Pinu", "Pice",... to 0, 1,...
counter = 0
tree_count_dict = {}
for i in data.iterrows():
    try:
        tree_count_dict[i[1]["tree_genus"]]
    except KeyError:
        tree_count_dict[i[1]["tree_genus"]] = counter
        counter += 1

data = data.copy().replace(to_replace=tree_count_dict)
data = data.copy().replace(to_replace=[1, 2, 3, 4], value=1)
print(data)

correlation = data.corr()
sns.heatmap(correlation, annot = True)
ds = data.to_numpy()

figure = plt.figure(figsize=(27, 9))
i = 1

# preprocess dataset, split into training and test part
# X, y = ds[:, (3,5)], ds[:, 5]

X, y = ds[:, (2, 3)], ds[:, 5]

# print(X.shape, y.shape)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.inferno
cm_bright = ListedColormap(['#0000FF','#F0FF00','#FF0000'])
ax = plt.subplot(1, len(classifiers) + 1, i)

ax.set_title("Input data")

# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

# Plot the testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print(name, ": Progress 1/3")
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train.astype('int'))
    score = clf.score(X_test, y_test)

    print(xx.shape, yy.shape, X_train.shape, y_train.shape)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    if i == 3:
        Z = Z.reshape(xx.shape[0], xx.shape[1])
    else:
        Z = Z.reshape(xx.shape[0], xx.shape[1])
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    print(name, ": Progress 2/3")

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    print(name, ": Progress 3/3")

    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()
