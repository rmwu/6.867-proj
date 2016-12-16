import numpy as np
import math

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def log_regression(data_train, data_validate, data_test,
                   lambdaa, threshold,
                   print_params=False):
    """
    Uses the SKLearn Logistic Regression module to perform logistic
    regression on our dataset.
    
    data_train     tuple of (X_train, y_train)
    data_validate  tuple of (X_verify, y_verify)
    data_test      tuple of (X_test, y_test)
    
    lambdaa        regularization?
    threshold      tolerance? I don't remember
    """
    # load testing data
    X_train = data_train[0]
    y_train = data_train[1]
    
    # regularization factor
    c = 1.0 / lambdaa
    
    l2reg = LogisticRegression(penalty="l2", C=c, tol=threshold)
    l1reg = LogisticRegression(penalty="l1", C=c, tol=threshold)
    
    # fit the data
    l2reg.fit(X_train, y_train)
    l1reg.fit(X_train, y_train)

    # load data from csv files
    X_validate = data_validate[0]
    y_validate = data_validate[1]
    
    print("Out-of-the-box scores.")
    print("L2 accuracy: {}".format(l2reg.score(X_validate, y_validate)))
    if print_params:
        print("L2 params {}".format(l2reg.coef_))
    print("L1 accuracy: {}".format(l1reg.score(X_validate, y_validate)))
    if print_params:
        print("L1 params {}".format(l1reg.coef_))
    print("")
    
def random_forest(data_train, data_validate, data_test,
                 n_estimators=10, entropy=True,
                 max_features=1.0, max_depth=None,
                 min_samples_leaf=1, min_impurity_split=1e-7, 
                 bootstrap=True,
                 oob_score=False,
                 warm_start=False,
                 verbose=False):
    """
    Uses the SKLearn Random Forest Classifier to perform
    bagged tree classification
    """
    # load data
    X_train = data_train[0]
    y_train = data_train[1]
    
    X_test = data_test[0]
    y_test = data_test[1]
    
    # set params
    criterion = "entropy" if entropy else "gini"
    
    # pass in all parameters
    randomForest = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        
        max_features=max_features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        
        bootstrap=bootstrap,
        oob_score=oob_score
    )
    
    leaves = randomForest.fit(X_train, y_train)
    feature_importance = randomForest.feature_importances_
    score = randomForest.score(X_test, y_test)
    
    print("Out of the box score {}".format(score))
    if verbose:
        print("Feature importances {}".format(feature_importance))

def multilayer_perceptron(data_train, data_test, hidden_layer_sizes, alpha=1e-5, activation='tanh', random_state=None):
    """
    alpha: L2 regularization parameter
    activation: can be 'identity', 'logistic', 'tanh', 'relu'
    random_state: seed for random number generator for MLPClassifier.
        Specify a fixed int to get deterministic behavior.
    """
    X_train, y_train = data_train
    X_test, y_test = data_test

    mlp = MLPClassifier(solver='adam', alpha=alpha, activation=activation,
        hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict_proba(X_test)
    cross_entropy_loss = sklearn.metrics.log_loss(y_test, y_pred, normalize=True)
    print("Cross entropy loss: {}".format(cross_entropy_loss))

def svm_classify(data_train, data_test, C, kernel='rbf', degree=3, balanced=True):
    """
    C: C-svm param
    kernel: the kernel
    degree: degree if we use "poly" kernel
    balanced: whether to weight the data different
    """
    X_train, y_train = data_train
    X_test, y_test = data_test

    print("Training SVM, C={}, kernel={}, balanced={} ...".format(C, kernel, balanced))
    svm_classifier = sklearn.svm.SVC(C=C, kernel=kernel, degree=degree, coef0=1.0, shrinking=True,
        tol=0.001, class_weight=('balanced' if balanced else None),
        verbose=False, decision_function_shape='ovr')
    svm_classifier.fit(X_train, y_train)
    accuracy = svm_classifier.score(X_test, y_test)
    print("SVM accuracy {}".format(accuracy))
    return accuracy
