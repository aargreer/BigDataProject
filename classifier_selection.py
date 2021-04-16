import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

h = .02  # step size in the mesh
names = ["Nearest Neighbors", "Decision Tree", "Random Forest",  "Neural Net", "Naive Bayes"]

classifiers = [ KNeighborsClassifier(5), DecisionTreeClassifier(max_depth=11),
                RandomForestClassifier(max_depth=11, n_estimators=20, max_features=2),
                MLPClassifier(alpha=1, max_iter=2000), GaussianNB() ]

data = pd.read_csv("data/fire_data_full.csv")

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

# outputs a heatmap with feature correlations
correlation = data.corr()
sns.heatmap(correlation, annot = True)

ds = data.to_numpy()
figure = plt.figure(figsize=(27, 9))
i = 1

# X is a vertical slice of a tuple of features, y is our classifier variable BURNCLAS
X, y = ds[:, (2,3)], ds[:, 5]

# preprocess dataset, split into training and test part
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# getting min and max values to determine mesh resolution
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# just plot the dataset first
colour_manager = plt.cm.inferno
cm_bright = ListedColormap(['#0000FF','#F0FF00','#FF0000'])
axis = plt.subplot(1, len(classifiers) + 1, i)

axis.set_title("Input data")

# Plot the training points
axis.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

# Plot the testing points
axis.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
axis.set_xlim(xx.min(), xx.max())
axis.set_ylim(yy.min(), yy.max())
axis.set_xticks(())
axis.set_yticks(())
i += 1

# iterate over classifiers
for name, classifier in zip(names, classifiers):
    print(name, ":\t\t Progress 1/3")
    axis = plt.subplot(1, len(classifiers) + 1, i)
    classifier.fit(X_train, y_train.astype('int'))

    # Plot the decision boundary; assigning a color to each point in the mesh [x_min, x_max] x [y_min, y_max].
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape[0], xx.shape[1])
    axis.contourf(xx, yy, Z, cmap=colour_manager, alpha=.8)

    print(name, ":\t\t Progress 2/3")

    # Plot the training points
    axis.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

    # Plot the testing points
    axis.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

    # setting bounds for subplots
    axis.set_xlim(xx.min(), xx.max())
    axis.set_ylim(yy.min(), yy.max())
    axis.set_xticks(())
    axis.set_yticks(())

    print(name, ":\t\t Progress 3/3\n")

    axis.set_title(name)
    accuracy = classifier.score(X_test, y_test)
    axis.text(xx.max() - .3, yy.min() + .3, ('%.2f' % accuracy).lstrip('0'), size=15, horizontalalignment='right')
    i += 1

# display output
plt.tight_layout()
plt.show()
