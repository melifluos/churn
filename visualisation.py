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
from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_iris
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)


def plot_multiclass_roc_curve(X, y):
    n_classes = len(np.unique(y))
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Binarize the output
    y = label_binarize(y, classes=range(n_classes))
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                         random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('local_results/multi_class_roc.png', bbox_inches='tight')
    plt.clf()


##############################################################################


def plot_roc(X, y, clf_class, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
    for algo_tuple in clf_class:
        name = algo_tuple[0]
        algo = algo_tuple[1]
        try:
            clf = algo(probability=True, **kwargs)
            clf.fit(X_train, y_train)
        except TypeError:
            clf = algo()
            clf.fit(X_train, y_train)

        preds = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=name + ' (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('local_results/roc.png')
    plt.clf()


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


def gen_x_vals(n_examples, n_vals=5):
    interval = int(n_examples / (n_vals + 2))
    return np.arange(interval, 6 * interval, interval)


def plot_learning_curve(X, y, algo, title='learning curve'):
    """
    Plot learning curves to diagnose bias and variance
    :param X: input values
    :param y: output values
    :param title: plot title
    :return:
    """

    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    x_vals = list(gen_x_vals(len(y)))
    print x_vals
    try:
        train_sizes, train_scores, valid_scores = learning_curve(algo, X, y, train_sizes=x_vals, cv=5)
    except ValueError:
        print "Probably because one of the splits contained just a single class - consider using bigger splits or " \
              "shuffling the indices to fix this problem"
        raise
    train_err = train_scores.std(axis=1)
    valid_err = valid_scores.std(axis=1)
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    plt.plot(train_sizes, train_mean, 'k', color='#CC4F1B', lw=2)
    plt.fill_between(train_sizes, train_mean + train_err, train_mean - train_err, alpha=0.5, edgecolor='#CC4F1B',
                     facecolor='#FF9848')
    plt.plot(train_sizes, valid_scores.mean(axis=1), 'k', color='#1B2ACC', lw=2)
    plt.fill_between(train_sizes, valid_mean + valid_err, valid_mean - valid_err, alpha=0.5, edgecolor='#CC4F1B',
                     facecolor='#1B2ACC')
    plt.legend(['Training Error', 'Test Error'])
    # plt.xscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Error')
    plt.title(title)
    plt.savefig('local_results/learning_curve_' + title + '.png', bbox_inches='tight')
    plt.clf()  # clear the current figure, but leave the window open


def run_test():
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

    plot_confusion_matrix(confusion_matrices, class_names)


def plot_confusion_matrix(confusion_matrices, labels, cmap=plt.cm.Blues):
    """
    Generates classification confusion matrix diagram(s)
    :param confusion_matrices: a list of tuples in the form (name, matrix)
    :param labels:
    :param cmap:
    :return:
    """
    plt.figure(1)
    for idx, mat in enumerate(confusion_matrices):
        plot_idx = 130 + int(idx) + 1
        plt.subplot(plot_idx)
        cm = mat[1]
        title = mat[0]
        cax = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        # plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # cbar = plt.colorbar(cax)
        # plt.delaxes(plt.axes)
        for x in xrange(cm.shape[1]):
            for y in xrange(cm.shape[0]):
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')
    plt.savefig('local_results/confusion_mat_' + title + '.png', bbox_inches='tight')
    plt.clf()


def plot_boxplots(data):
    """

    :param data:
    :return:
    """
    all_data = data.values
    labels = data.columns
    plt.boxplot(all_data)
    plt.set_title('box plot')
    plt.yaxis.grid(True)
    plt.set_xticks([y + 1 for y in range(len(all_data))])
    plt.set_xlabel('feature')
    plt.set_ylabel('ylabel')
    plt.setp(xticks=[y + 1 for y in range(len(all_data))], xticklabels=labels)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    customers = pd.read_csv('local_resources/customer/000000_0', sep='\t')
    cust_columns = ['id', 'churn', 'gender', 'country', 'created_on', 'yob', 'premier']
    customers.columns = cust_columns
    customers.set_index('id', inplace=True)
    customers['churn'] -= 1
    #  sample some data
    rows = np.random.choice(customers.index.values, 3000)
    customers = customers.ix[rows]
    # Isolate target data
    y = np.array(customers['churn'])
    # We don't need these columns
    to_drop = ['churn', 'created_on', 'country', 'gender']
    churn_feat_space = customers.drop(to_drop, axis=1)
    features = churn_feat_space.columns.values
    X = churn_feat_space.as_matrix().astype(np.float)
    # print y.shape
    # print y

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    algos = [('SVM', SVC), ('RF', RF), ('KNN', KNN)]
    plot_roc(X, y, algos)
