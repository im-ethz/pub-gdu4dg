import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd


from sklearn import svm
from sklearn import metrics
from Model.utils import transform_one_hot, decode_one_hot_vector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier



def svm_classification(train_embedding, train_labels, val_embedding, val_labels, multi_class=False):

    svm_clf = svm.SVC(kernel="poly").fit(train_embedding, train_labels.ravel())

    val_pred = np.expand_dims(svm_clf.predict(val_embedding), 1)

    svm_accuracy = metrics.accuracy_score(val_labels, val_pred)


    if multi_class is False:
        svm_auc = metrics.roc_auc_score(val_labels, val_pred)
        svm_f1 = metrics.f1_score(val_labels, val_pred)
    else:
        svm_clf_prob = svm.SVC(kernel="poly", decision_function_shape='ovo', probability=True ).fit(train_embedding, train_labels.ravel())
        svm_auc = metrics.roc_auc_score(transform_one_hot(val_labels), svm_clf_prob.predict_proba(val_embedding), multi_class='ovo', average='macro')
        svm_f1 = metrics.f1_score(val_labels, val_pred, average='micro')

    return svm_accuracy, svm_auc, svm_f1


def logistic_classification(train_embedding, train_labels, val_embedding, val_labels, multi_class=False):

    logi_clf = LogisticRegression(random_state=0).fit(train_embedding, train_labels.ravel())

    val_pred = logi_clf.predict(val_embedding)

    logi_accuray = metrics.accuracy_score(val_labels, val_pred)

    if multi_class is False:
        logi_auc = metrics.roc_auc_score(val_labels, val_pred)
        logi_f1 = metrics.f1_score(val_labels, val_pred)
    else:
        logi_auc = metrics.roc_auc_score(transform_one_hot(val_labels), logi_clf.predict_proba(val_embedding), multi_class='ovo', average='macro')
        logi_f1 = metrics.f1_score(val_labels, val_pred, average='micro')

    return logi_accuray, logi_auc, logi_f1


def gradient_boosting_classification(train_embedding, train_labels, val_embedding, val_labels, multi_class=False):

    gb_clf = GradientBoostingClassifier(random_state=0).fit(train_embedding, train_labels.ravel())

    val_pred = gb_clf.predict(val_embedding)

    gb_accuracy = metrics.accuracy_score(val_labels, val_pred)


    if multi_class is False:
        gb_auc = metrics.roc_auc_score(val_labels, val_pred)
        gb_f1 = metrics.f1_score(val_labels, val_pred)
    else:
        gb_auc = metrics.roc_auc_score(transform_one_hot(val_labels), gb_clf.predict_proba(val_embedding), multi_class='ovo', average='macro')
        gb_f1 = metrics.f1_score(val_labels, val_pred, average='micro')

    return gb_accuracy, gb_auc, gb_f1


def model_evaluation(train_embedding, train_labels, val_embedding, val_labels, multi_class=False):

    if multi_class:
        # transform one hot into multilabaled vecotor
        train_labels = decode_one_hot_vector(train_labels)
        val_labels = decode_one_hot_vector(val_labels)


    svm_accuracy, svm_auc, svm_f1 = svm_classification(train_embedding, train_labels, val_embedding, val_labels, multi_class=multi_class)
    logistic_accuray, logistic_auc, logistic_f1 = logistic_classification(train_embedding, train_labels, val_embedding, val_labels, multi_class=multi_class)
    gb_accuracy, gb_auc, gb_f1 = gradient_boosting_classification(train_embedding, train_labels, val_embedding, val_labels, multi_class=multi_class)

    accuracy_df = pd.DataFrame([svm_accuracy, logistic_accuray, gb_accuracy], columns=['accuracy']).round(5)
    auc_df = pd.DataFrame([svm_auc, logistic_auc, gb_auc], columns=['auc']).round(5)
    f1_df = pd.DataFrame([svm_f1, logistic_f1, gb_f1], columns=['f1']).round(5)

    eval_df = pd.concat([accuracy_df, auc_df, f1_df], axis=1)
    eval_df.index = ['SVM', 'LogReg', 'GradBoost']

    return eval_df
