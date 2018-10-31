import argparse
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import pickle

from helpers import compute_AUC_scores, construct_kaggle_submissions, plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="XGBoost", help="Model")
parser.add_argument("--remarks", "-r", type=str, default="", help="Remarks")
parser.add_argument("--balanced", type=bool, default=False, help="Drop useless features")
# Booleans (False by default)
parser.add_argument("--scale", action='store_true', help="Scale data")
parser.add_argument("--drop", action='store_true', help="Drop useless features")
args = parser.parse_args()

train_data = pd.read_csv("data/train_data.csv", header=None).values
test_data = pd.read_csv("data/test_data.csv", header=None).values
train_labels = pd.read_csv("data/train_labels.csv", header=None,
                              names=['class']).values.ravel()
if args.drop:
    train_data_df = pd.read_csv("data/train_data.csv", header=None)
    test_data_df = pd.read_csv("data/test_data.csv", header=None)
    train_data = train_data_df.drop([0, 216, 217, 218, 219], axis=1).values
    test_data = test_data_df.drop([0, 216, 217, 218, 219], axis=1).values

if args.scale:
    # Standardize
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = min_max_scaler.fit_transform(train_data)
    test_data = min_max_scaler.fit_transform(test_data)

train_data, eval_data, train_labels, eval_labels = \
    train_test_split(train_data, train_labels, random_state=7, test_size=0.3)

eval_set = [(train_data, train_labels), (eval_data, eval_labels)]

if args.model == "XGBoost":
    model = XGBClassifier(learning_rate=0.1, max_depth=5,
                      min_child_weight=7, n_estimators=100, nthread=1, subsample=0.7,
                      objective='multi:softprob', num_class=10)
    model.fit(train_data, train_labels,
              eval_metric=["merror", "mlogloss"], eval_set=eval_set,
              verbose=True)
    results_proba = model.predict_proba(test_data)
    eval_predicted_proba = model.predict_proba(eval_data)
elif args.model == "RandomForest":
    model = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.4, min_samples_leaf=4, min_samples_split=12, n_estimators=100, verbose=3, class_weight='balanced')
    model.fit(train_data, train_labels)
    results_proba = model.predict_proba(test_data)
    eval_predicted_proba = model.predict_proba(eval_data)
elif args.model == "SVC":
    model = LinearSVC(C=1.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.01, verbose=3)
    model.fit(train_data, train_labels)
    results_proba = model.decision_function(test_data)
    eval_predicted_proba = model.decision_function(eval_data)

results = model.predict(test_data)  # Predicts from 1-10
eval_predicted = model.predict(eval_data)

if args.model != "SVC":  # SVC doesn't implement predict_proba so CV is not possible
    kfold = StratifiedKFold(n_splits=3, random_state=7)
    scores = cross_val_score(model, train_data, train_labels,
                             cv=kfold, scoring="neg_log_loss")
    accuracy_scores = cross_val_score(model, train_data, train_labels,
                             cv=kfold, scoring="accuracy")
    print("Cross validation logloss scores: {:.5f} {:.5f}".format(np.mean(scores),
                                                                  np.std(scores)))
    print("Cross validation accuracy scores: {:.5f} {:.5f}".format(np.mean(accuracy_scores),
                                                                   np.std(accuracy_scores)))

onehot = to_categorical(eval_labels).astype(int) # Splits into classes from 0-10 (11 classes)
eval_onehot = onehot[:, 1:]  # Trim unnecessary first column (class "0")
ll = log_loss(eval_onehot, eval_predicted_proba)
acc = accuracy_score(eval_labels, eval_predicted)
print("Validation log-loss and accuracy: {:.5f} {:.5f}".format(ll, acc))

uid = "{}_scaled={}_drop={}_remarks={}".format(args.model, args.scale, args.drop, args.remarks)

########## Plot
if args.model in ["XGBoost"]:
    train_metrics = model.evals_result()['validation_0']
    test_metrics = model.evals_result()['validation_1']
    epochs = len(train_metrics['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_metrics['mlogloss'], label='Train')
    ax.plot(x_axis, test_metrics['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('{} - Log Loss'.format(args.model))
    plt.savefig("img/logloss_{}.png".format(uid))
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_metrics['merror'], label='Train')
    ax.plot(x_axis, test_metrics['merror'], label='Test')
    ax.legend()
    plt.ylabel('Error')
    plt.title('{} - Error'.format(args.model))
    plt.savefig("img/error_{}.png".format(uid))
    plt.show()

# Save Kaggle submission files
construct_kaggle_submissions(uid, results, results_proba)

model_file = open("models/{}.mdl".format(uid), "wb")
pickle.dump(model, model_file)

# Confusion matrix
plot_confusion_matrix(eval_predicted, eval_labels, args.model, uid)

# AUC ROC scores
compute_AUC_scores(eval_predicted_proba, eval_labels)
