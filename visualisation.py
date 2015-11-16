from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
# This is important
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)


def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = StratifiedKFold(y, n_folds=5)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob


churn_df = pd.read_csv('local_resources/telco_churn.csv')
col_names = churn_df.columns.tolist()

print "Column names:"
print col_names

to_show = col_names[:6] + col_names[-6:]

print "\nSample data:"
print churn_df[to_show].head(6)

# Isolate target data
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.', 1, 0)

# We don't need these columns
to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_feat_space = churn_df.drop(to_drop, axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan", "VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)

y = np.array(y)
class_names = np.unique(y)

confusion_matrices = [
    ("SVM", confusion_matrix(y, run_cv(X, y, SVC))),
    ("RF", confusion_matrix(y, run_cv(X, y, RF))),
    ("KNN", confusion_matrix(y, run_cv(X, y, KNN))),
]


def plot_confusion_matrix(confusion_matrices, labels, cmap=plt.cm.Blues):
    plt.figure(1)
    for idx, mat in enumerate(confusion_matrices):
        plot_idx = 130 + int(idx)
        plt.subplot(plot_idx)
        cm = mat[1]
        title = mat[0]
        cax = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        cbar = plt.colorbar(cax)
        # plt.delaxes(plt.axes)
        for x in xrange(2):
            for y in xrange(2):
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')
    plt.show()

plot_confusion_matrix(confusion_matrices, class_names)

# fig, ax = plt.subplots(ncols=3)
# for idx, axis in enumerate ax:
#
#     axis.pcolor(matrix[0][1], cmap=plt.cm.Blues)
#     ax_rf.pcolor(matrix[1][1], cmap=plt.cm.Blues)
#     ax_knn.pcolor(matrix[2][1], cmap=plt.cm.Blues)
#
#     # put the major ticks at the middle of each cell
#     # ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
#     # ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
#     #
#     # # want a more natural, table-like display
#     # ax.invert_yaxis()
#     # ax.xaxis.tick_top()
#     #
#     # ax.set_xticklabels(row_labels, minor=False)
#     # ax.set_yticklabels(column_labels, minor=False)
#     plt.show()
#
#
# plot_confusion(confusion_matrices, class_names)
