import numpy as np
import math

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def log_regression(data_train, data_validate, data_test,
                   lambdaa, penalty, threshold,
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
    c = 1.0 / lambdaa
    
    log_reg = LogisticRegression(penalty=penalty, C=c, tol=threshold)
    
    # fit the data
    log_reg.fit(*data_train)

    dataz = [data_train, data_validate, data_test]
    accuracies = [log_reg.score(*dataset) for dataset in dataz]
    print("Train/val/test accuracies: {}".format(accuracies))

    cross_entropies = [sklearn.metrics.log_loss(dataset[1], log_reg.predict_proba(dataset[0]), normalize=True) for dataset in dataz]
    print("t/v/t cross-entropy loss: {}".format(cross_entropies))
    return (log_reg, accuracies, cross_entropies)
    
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
    
    scoreTrain = randomForest.score(X_train, y_train)
    scoreTest = randomForest.score(X_test, y_test)
    
    print("Training score {}".format(scoreTrain))
    print("Testing score {}".format(scoreTest))
    if verbose:
        print("Feature importances {}".format(feature_importance))
        
def adaboost(data_train, data_test,
                 n_estimators=50, learning_rate=1.,
                 base_estimator=None,
                 verbose=False):
    """
    n_estimators    number of estimators
    learning_rate   contribution of each classifier 
                    (tradeoff w/ n_estimators)
    """
    adaB = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        base_estimator=base_estimator
    )
    
    # load data
    X_train = data_train[0]
    y_train = data_train[1]
    
    X_test = data_test[0]
    y_test = data_test[1]
    
    adaB.fit(X_train, y_train)
    
    score_train = adaB.score(X_train, y_train)
    score_test = adaB.score(X_test, y_test)
    
    print("Train accuracies {}".format(score_train))
    print("Test accuracies {}".format(score_test))
    
def gradboost(data_train, data_test,
                 n_estimators=50, learning_rate=1.,
                 verbose=False):
    """
    n_estimators    number of estimators
    learning_rate   contribution of each classifier 
                    (tradeoff w/ n_estimators)
    """
    adaB = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    
    # load data
    X_train = data_train[0]
    y_train = data_train[1]
    
    X_test = data_test[0]
    y_test = data_test[1]
    
    adaB.fit(X_train, y_train)
    
    score_train = adaB.score(X_train, y_train)
    score_test = adaB.score(X_test, y_test)
    
    print("Train accuracies {}".format(score_train))
    print("Test accuracies {}".format(score_test))

def multilayer_perceptron(data_train, data_val, data_test, hidden_layer_sizes,
                          alpha=1e-5, activation='tanh', random_state=None):
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
    mlp.fit(*data_train)

    dataz = [data_train, data_val, data_test]
    accuracies = [mlp.score(*dataset) for dataset in dataz]
    print("t/v/t accuracy: {}".format(accuracies))

    cross_entropies = [sklearn.metrics.log_loss(dataset[1], mlp.predict_proba(dataset[0]), normalize=True)
      for dataset in dataz]
    print("t/v/t cross-entropy loss: {}".format(cross_entropies))

    return (mlp, accuracies, cross_entropies)

def svm_classify(data_train, data_val, data_test, C, kernel='rbf', degree=3, balanced=True):
    """
    C: C-svm param
    kernel: the kernel
    degree: degree if we use "poly" kernel
    balanced: whether to weight the data different
    """
    X_train, y_train = data_train
    X_test, y_test = data_test

    dataz = (data_train, data_val, data_test)

    print("Training SVM, C={}, kernel={}, balanced={} ...".format(C, kernel, balanced))
    svm_classifier = sklearn.svm.SVC(C=C, kernel=kernel, degree=degree, coef0=1.0, shrinking=True,
        tol=0.001, class_weight=('balanced' if balanced else None),
        verbose=False, decision_function_shape='ovr')
    svm_classifier.fit(*data_train)
    
    accuracies = get_accuracies(svm_classifier, dataz)
    cross_entropies = get_cross_entropies(svm_classifier, dataz)
    return (svm_classifier, accuracies, cross_entropies)

def sgd_svm_classify(data_train, data_val, data_test, alpha):
    dataz = (data_train, data_val, data_test)

    print("Training SVM, alpha={} ...".format(alpha))
    sgd_svm = sklearn.linear_model.SGDClassifier(alpha=alpha)
    sgd_svm.fit(*data_train)

    accuracies = get_accuracies(svm_classifier, dataz)
    cross_entropies = get_cross_entropies(svm_classifier, dataz)
    return (sgd_svm, accuracies, cross_entropies)

def get_accuracies(classifier, dataz):
    accuracies = [classifier.score(*dataset) for dataset in dataz]
    print("tvt accuracies {}".format(accuracies))
    return accuracies

def get_cross_entropies(classifier, dataz):
    cross_entropies =  [sklearn.metrics.log_loss(dataset[1], classifier.predict_proba(dataset[0]), normalize=True)
      for dataset in dataz]
    print("tvt x-entrop {}".format(cross_entropies))
    return cross_entropies