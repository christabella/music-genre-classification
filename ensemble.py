import argparse
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from helpers import compute_AUC_scores, construct_kaggle_submissions, plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="ensemble_rf_xgb", help="Model")
parser.add_argument("--final", action='store_true', help="Put all training data into model fit")
args = parser.parse_args()


def load_model(filename):
    return pickle.load(open(filename, 'rb'))


RF_model = load_model('models/RandomForest_scaled=True_drop=False_remarks=Unweighted.mdl')
XGB_model = load_model('models/XGBoost_scaled=False_n=100.mdl')
SVC_model = load_model('models/SVC_scaled=True_drop=False_remarks=c=5.mdl')
final_RF_model = load_model('models/FINAL_RandomForest_scaled=True_drop=False_remarks=5.mdl')
final_XGB_model = load_model('models/FINAL_XGBoost_scaled=False_drop=False_remarks=5.mdl')
final_SVC_model = load_model('models/FINAL_SVC_scaled=True_drop=False_remarks=5.mdl')

unfitted_RF_model = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.4,
                                           min_samples_leaf=4, min_samples_split=12,
                                           n_estimators=100, verbose=3, class_weight='balanced')
unfitted_XGB_model = XGBClassifier(learning_rate=0.1, max_depth=5,
                                   min_child_weight=7, n_estimators=100, nthread=1, subsample=0.7,
                                   objective='multi:softprob', num_class=10)
unfitted_SVC_model = CalibratedClassifierCV(LinearSVC(C=5.0, dual=False, loss="squared_hinge",
                                                      penalty="l1", tol=1e-4, verbose=3))

ensembles = {
    'ensemble_rf_xgb': {
        'uid' : 'ensemble_rf_xgb',
        'estimators': [('RF', RF_model), ('XGB', XGB_model)],
        'title': 'VotingClassifier: XGB, RF'
    },
    'ensemble_rf_xgb_svc': {
        'uid' : 'ensemble_rf_xgb_svc',
        'estimators': [('RF', RF_model), ('XGB', XGB_model), ('SVC', SVC_model)],
        'title': 'VotingClassifier: XGB, RF, SVC'
    },
    'final_ensemble_rf_xgb': {
        'uid' : 'final_ensemble_rf_xgb',
        'estimators': [('RF', final_RF_model), ('XGB', final_XGB_model)],
        'title': 'VotingClassifier: XGB, RF (Final)'
    },
    'final_ensemble_rf_xgb_svc': {
        'uid' : 'final_ensemble_rf_xgb_svc',
        'estimators': [('RF', final_RF_model), ('XGB', final_XGB_model), ('SVC', final_SVC_model)],
        'title': 'VotingClassifier: XGB, RF, SVC (Final)'
    },
    'ensemble_rf_xgb_unfitted': {
        'uid' : 'ensemble_rf_xgb_unfitted',
        'estimators': [('RF', unfitted_RF_model),
                     ('XGB', unfitted_XGB_model)],
        'title': 'VotingClassifier: XGB, RF (unfitted)'
    },
    'ensemble_rf_xgb_svc_unfitted': {
        'uid' : 'ensemble_rf_xgb_unfitted',
        'estimators': [('RF', unfitted_RF_model),
                       ('XGB', unfitted_XGB_model),
                       ('SVC', unfitted_SVC_model)],
        'title': 'VotingClassifier: XGB, RF, SVC (unfitted)'
    }
}

ensemble = ensembles[args.model]
'''weights : array-like, shape = [n_classifiers]
Sequence of weights (float or int) to weight the occurrences of class probabilities before averaging (soft voting). Uses uniform weights if None.
'''
model = VotingClassifier(estimators=ensemble['estimators'], voting='soft', n_jobs=4, weights=None)

train_data = pd.read_csv("data/train_data.csv", header=None).values
test_data = pd.read_csv("data/test_data.csv", header=None).values
train_labels = pd.read_csv("data/train_labels.csv", header=None,
                              names=['class']).values.ravel()

if not args.final:
    train_data, eval_data, train_labels, eval_labels = \
        train_test_split(train_data, train_labels, random_state=7, test_size=0.3)

model = model.fit(train_data, train_labels)

kfold = StratifiedKFold(n_splits=3, random_state=7)
scores = cross_val_score(model, train_data, train_labels,
                         cv=kfold, scoring="neg_log_loss")
accuracy_scores = cross_val_score(model, train_data, train_labels,
                         cv=kfold, scoring="accuracy")
print("Cross validation logloss scores: {:.5f} {:.5f}".format(np.mean(scores), np.std(scores)))
print("Cross validation accuracy scores: {:.5f} {:.5f}".format(np.mean(accuracy_scores), np.std(accuracy_scores)))


# Save Kaggle submission files
results = model.predict(test_data)  # Predicts from 1-10
results_proba = model.predict_proba(test_data)
construct_kaggle_submissions(ensemble['uid'] + "_Final=" + str(args.final), results, results_proba)

model_file = open("models/{}.mdl".format(ensemble['uid'] + "_Final=" + str(args.final)), "wb")
pickle.dump(model, model_file)

if not args.final:
    # Confusion matrix
    eval_predicted = model.predict(eval_data)
    plot_confusion_matrix(eval_predicted, eval_labels, ensemble['title'], ensemble['uid'])

    # AUC ROC scores
    eval_predicted_proba = model.predict_proba(eval_data)
    compute_AUC_scores(eval_predicted_proba, eval_labels)
