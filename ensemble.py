import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

RF_model = pickle.load(open('models/RandomForest_scaled=True_drop=False_remarks=Unweighted.mdl', 'rb'))
XGB_model = pickle.load(open('models/XGBoost_scaled=False_n=100.mdl', 'rb'))

ensemble_rf_xgb = {
    uid : 'ensemble_rf_xgb',
    estimators: [('RF', RF_model), ('XGB', XGB_model)],
    title: 'VotingClassifier: XGB, RF'
}
ensemble = ensemble_rf_xgb
model = VotingClassifier(estimators=ensemble.estimators, voting='soft', n_jobs=4)

train_data = pd.read_csv("data/train_data.csv", header=None).values
test_data = pd.read_csv("data/test_data.csv", header=None).values
train_labels = pd.read_csv("data/train_labels.csv", header=None,
                              names=['class']).values.ravel()

train_data, eval_data, train_labels, eval_labels = \
    train_test_split(train_data, train_labels, random_state=7, test_size=0.3)

eval_set = [(train_data, train_labels), (eval_data, eval_labels)]


model = model.fit(train_data, train_labels)

kfold = StratifiedKFold(n_splits=3, random_state=7)
scores = cross_val_score(model, train_data, train_labels,
                         cv=kfold, scoring="neg_log_loss")
accuracy_scores = cross_val_score(model, train_data, train_labels,
                         cv=kfold, scoring="accuracy")
print("Cross validation logloss scores: {:.5f} {:.5f}".format(np.mean(scores), np.std(scores)))
print("Cross validation accuracy scores: {:.5f} {:.5f}".format(np.mean(accuracy_scores), np.std(accuracy_scores)))


# Save Kaggle submission files
construct_kaggle_submissions(ensemble.uid, results, results_proba)

model_file = open("models/{}.mdl".format(ensemble.uid), "wb")
pickle.dump(model, model_file)

# Confusion matrix
eval_predicted = model.predict(eval_data)
plot_confusion_matrix(eval_predicted, eval_labels, ensemble.title, ensemble.uid)

# AUC ROC scores
eval_predicted_proba = model.predict_proba(eval_data)
compute_AUC_scores(eval_predicted_proba, eval_labels)
