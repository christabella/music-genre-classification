import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

train_data = pd.read_csv("data/train_data.csv", header=None).values
test_data = pd.read_csv("data/test_data.csv", header=None).values
train_labels = pd.read_csv("data/train_labels.csv", header=None,
                              names=['class']).values.ravel()

# Average CV score on the training set was:-1.1047596381471787
model = XGBClassifier(learning_rate=0.1, max_depth=5,
                      min_child_weight=7, n_estimators=1000, nthread=1, subsample=0.7,
                      objective='multi:softmax', num_class=10)

train_data, eval_data, train_labels, eval_labels = \
            train_test_split(train_data, train_labels, random_state=7)

eval_set = [(eval_data, eval_labels)]
model.fit(train_data, train_labels,
          eval_metric="mlogloss", eval_set=eval_set,
          verbose=True)
scores = cross_val_score(model,
                         train_data, train_labels,
                         cv=7, scoring="neg_log_loss")
print("Cross validation scores:", scores)

results = model.predict_proba(test_data)
# Label each row to match Kaggle submission format
indices = np.arange(1, results.shape[0] + 1)
labeled_results = np.append(indices[:,None], results, axis=1)

np.savetxt("data/logloss_xgboost.csv", labeled_results, fmt="%d" + 10 * ", %f",
           header="Sample_id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,Class_10",
           comments='')
