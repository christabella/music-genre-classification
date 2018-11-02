import seaborn as sns
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

### SVC classifier
model = LinearSVC(loss="squared_hinge", tol=1e-4, verbose=0, dual=False)

svc_param_grid = {'penalty': ['l1', 'l2'],
                  'C': [1, 5, 10, 15, 50, 100, 1000]}

kfold = StratifiedKFold(n_splits=3, random_state=7)

gsSVMC = GridSearchCV(model,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose=3)
# gsSVMC = GridSearchCV(model,param_grid = svc_param_grid, cv=kfold, scoring="neg_log_loss", n_jobs= 4, verbose=3)

train_data = pd.read_csv("data/train_data.csv", header=None).values
test_data = pd.read_csv("data/test_data.csv", header=None).values
train_labels = pd.read_csv("data/train_labels.csv", header=None,
                              names=['class']).values.ravel()

gsSVMC.fit(train_data, train_labels)

model= 'gridsearch_SVM'
SVMC_best = gsSVMC.best_estimator_
model_file = open("models/{}.mdl".format(model), "wb")
pickle.dump(SVMC_best, model_file)

# Best score
print("Best score: ", gsSVMC.best_score_)

pickle.dump(gsSVMC, open("gridsearches/gsSVM", "wb"))

scores = np.array(gsSVMC.cv_results_['mean_test_score']).reshape(7, 2).T

plt.figure(figsize=(10, 4))
sns.heatmap(scores, xticklabels=gsSVMC.param_grid['C'],
            yticklabels=gsSVMC.param_grid['penalty'],
            vmin=np.min(scores), vmax=np.max(scores), annot=scores, fmt=".5f")
plt.xlabel("C (penalty parameter)")  # Smaller C means
plt.ylabel("penalty loss)")  # Smaller C means
plt.title("Accuracies from grid search on LinearSVC")
plt.savefig("img/gridsearch_{}.png".format(model))
plt.show()
