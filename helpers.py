from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.utils import to_categorical

def construct_kaggle_submissions(uid, results, results_proba):
    ########## Submission
    # For logloss submission
    # Label each row to match Kaggle submission format
    indices = np.arange(1, results_proba.shape[0] + 1)
    labeled_results_proba = np.append(indices[:,None], results_proba, axis=1)

    np.savetxt("submissions/logloss_{}.csv".format(uid),
               labeled_results_proba, fmt="%d" + 10 * ", %f",
               header="Sample_id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,Class_10",
               comments='')

    # For accuracy submission
    # Label each row to match Kaggle submission format
    indices = np.arange(1, results.shape[0] + 1)
    labeled_results = np.append(indices[:,None], results[:,None], axis=1)
    np.savetxt("submissions/accuracy_{}.csv".format(uid),
               labeled_results, delimiter=",", fmt="%d", 
               header="Sample_id,Sample_label",
               comments='')

def plot_confusion_matrix(eval_predicted, eval_labels, model, uid):
    cm = confusion_matrix(eval_labels, eval_predicted)
    # Get proportions per true class (row)
    sums_per_row = cm.sum(axis=1)
    sums_matrix = np.repeat(sums_per_row[None, :], 10, axis=0).T
    cm_proportions = cm/sums_matrix
    heatmap = sns.heatmap(cm_proportions, cmap="Blues", annot=cm, fmt="d",
                          vmin=0, vmax=1)
    fontsize=14
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('{} - Confusion matrix'.format(model))
    plt.savefig("img/cm_{}.png".format(uid))
    plt.show()

def compute_AUC_scores(eval_predicted_proba, eval_labels):
    onehot = to_categorical(eval_labels).astype(int) # Splits into classes from 0-10 (11 classes)
    eval_onehot = onehot[:, 1:]  # Trim unnecessary first column (class "0")
    weighted_auc = roc_auc_score(eval_onehot, eval_predicted_proba, average='weighted')
    macro_auc = roc_auc_score(eval_onehot, eval_predicted_proba, average='macro')
    print("Weighted AUC: {:.5f}, Macro AUC: {:.5f}".format(weighted_auc, macro_auc))
