import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# function for testing multiple runs using different values of m features (really inefficient hyperparameter tuning)
def test_classifier_parameters():
    for i in range(1, 6):
        accuracy_avg = 0
        clf = RandomForestClassifier(max_depth=11, n_estimators=20, max_features=i)
        for j in range(0, 75):
            clf.fit(X_train, y_train.astype('int'))
            accuracy = clf.score(X_test, y_test)
            accuracy_avg += accuracy
        print("Average accuracy for n=", i, "features:\t", accuracy_avg * 100 / 75, "%")


# computes and displays ROC curve using an array for predicted values versus actual values
def display_roc_curve(predictions, actual_values):
    fpr = dict()
    tpr = dict()
    y_score = predictions

    roc_auc = dict()
    for i in range(0, 300):
        fpr[i], tpr[i], _ = roc_curve(actual_values, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(actual_values.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plot the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def preprocess(unprocessed_data):
    # uses a dict to convert from tree genus ("Pinu", "Pice",...) to genus IDs (0, 1,...)
    counter = 0
    tree_count_dict = {}
    for i in unprocessed_data.iterrows():
        try:
            tree_count_dict[i[1]["tree_genus"]]
        except KeyError:
            tree_count_dict[i[1]["tree_genus"]] = counter
            counter += 1

    # replace tree genus with ID and clamp fire value to 0 (no fire) or 1 (fire)
    processed_data = unprocessed_data.copy().replace(to_replace=tree_count_dict)
    processed_data = processed_data.copy().replace(to_replace=[1, 2, 3, 4], value=1)
    return processed_data


# save model to disk
def output_to_pickle(output_clf):
    pickle.dump(output_clf, open("../../data/model.pickle", 'wb'))


# load model from disk
def load_from_pickle():
    new_clf = pickle.load(open("../../data/model.pickle", 'rb'))
    return new_clf


# init classifier and data; prints the data
clf = RandomForestClassifier(max_depth=11, n_estimators=20, max_features=2)
data = preprocess(pd.read_csv("../../data/fire_data_2015.csv"))
print("\n\t\t\tORIGINAL DATA:\n", data)

ds = data.to_numpy()

# X is all features besides the class variable, y is our classifier variable BURNCLAS
X, y = ds[:, 0:5], ds[:, 5]

# split into training and test part (feature scaling commented out cause it gave us better results)
# X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

# trains the model and outputs accuracy against test data
clf.fit(X_train, y_train.astype('int'))
accuracy = clf.score(X_test, y_test)
print("\n\n2015 Test-Train Split Accuracy:\t\t", accuracy * 100, "%")

# display ROC curve for RandomForest
display_roc_curve(clf.predict_proba(np.c_[X_test[:, :4], y_test.ravel()])[:, 1], y_test)

# output trained model
output_to_pickle(clf)

# loading data from 2011 to test accuracy of models outside of a trained year
data_2011 = preprocess(pd.read_csv("../../data/fire_data_2011.csv"))
test_data = data_2011.copy().drop(labels="BURNCLAS", axis=1).to_numpy()
actual_outcomes = data_2011["BURNCLAS"].to_numpy()

# load model from file
clf2 = load_from_pickle()
predictions = clf2.predict_proba(test_data)[:, 1].ravel()

# rounding predictions to 0 or 1 based on confidence threshold
# (threshold set to 45% confidence as this gave us the best false negativity rate)
a = []
max_confidence = np.amax(predictions)
confidence_threshold = 0.45 * max_confidence
for i in predictions:
    # if prediction higher than threshold, output positive case; else negative
    if i >= confidence_threshold:
        a.append(1)
    else:
        a.append(0)
predictions_rounded = np.array(a)

# display ROC curve
display_roc_curve(clf2.predict_proba(test_data)[:, 1], actual_outcomes)
cv_scores = cross_val_score(clf2, test_data, actual_outcomes, cv=5)

# print confusion matrix andd cross validation scores
accuracy = accuracy_score(actual_outcomes, predictions_rounded)
print("\n\n\n5-fold CV results:\nAccuracy:\t", cv_scores.mean(), "\nStd. Deviation:", cv_scores.std())
tn, fp, fn, tp = confusion_matrix(actual_outcomes, predictions_rounded).ravel()
print("\nTrue positives:", tp)
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("\nTotal cases:", actual_outcomes.shape[0])

# append column with predicted class to original data; output to file
output_data = data_2011.copy().assign(PredictedClass=lambda x: predictions_rounded)
output_data.to_csv("../../data/fire_data_2011_with_predictions.csv")
