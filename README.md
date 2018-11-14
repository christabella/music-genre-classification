# music-genre-classification
## Abstract
To tackle the problem of classifying the musical genre of songs based on "carefully chosen features", approaches using **support vector machines (SVM), Random Forest, XGBoost, and deep neural networks** were used. Data analysis showed high class imbalance in the dataset, and visualizations such as the correlation matrix of features, 3-dimensional PCA projection, and feature-class plots, indicate that many of the features are possibly non-informative and that classes are not easily separable. The class imbalance was not addressed directly in our final approach, since the Kaggle test data is similarly imbalanced and attempts to balance the classes gave worse results on the validation dataset. **Feature scaling**, on the other hand, improved most of the models' performance. On the Kaggle dataset, the best variants of all models ranged from **[60.2%-66.2% in accuracy](https://kaggle.com/c/mlbp-data-analysis-challenge-accuracy-2018) and [0.236-0.171 in log-loss](https://www.kaggle.com/c/mlbp-data-analysis-challenge-log-loss-2018)**, with XGBoost scoring the highest on both metrics.

## Project Structure
The main report can be found in `Report.ipynb` and it describes the end-to-end machine learning workflow including exploratory data visualizations, feature preprocessing, model selection, experiments and results, and analysis.

###  Running Report notebook
Required Python packages are `seaborn, numpy, scikit-learn, matplotlib, pandas`

### Training `sklearn` models
Running `python classifiers.py` will train and evaluate a given model, construct Kaggle submissions, and save various plots (e.g. confusion matrix, learning curve) and pickled models to files.

Args:
  `--model`, `-m`: Can be XGBoost (default), RandomForest, SVC, or KNN.
 `--scale`: Scale data with min-max scaling.
 `--drop`: Drop non-informative MFCC features (216-219).
 `--final`: Use all training data when training the model (default=False)
 `--remarks`, -r: [optional] A string to be saved in the filenames of exported resources.
 `--neighbors`. If the model is KNN, an integer defining the number of neighbours.
An example usage is `python classifiers.py --model "SVC" --scale --drop --remarks "early stopping"`

### Training ensembles of `sklearn` models
Voting classifiers ensembling several models can be run with `python ensemble.py`, with arguments:
  `--model, -m`:  `ensemble_rf_xgb` (default), `ensemble_rf_xgb_svc`, `final_ensemble_rf_xgb`, `final_ensemble_rf_xgb_svc`, `ensemble_rf_xgb_unfitted`, `ensemble_rf_xgb_svc_unfitted`.
  `--final`: Use all training data when training the model.
  
An example usage is --model "SVC" --scale --drop --remarks "early stopping"
Args:
  --model, -m: Can be XGBoost (default), RandomForest, SVC, or KNN.
  --remarks, -r: A string to be saved in the filenames of exported resources.
  --neighbors. If the model is KNN, an integer defining the number of neighbours.
  --scale: Scale data with min-max scaling.
  --drop: Drop non-informative MFCC features (216-219).
  --final: Use all training data when training the model.

### Running grid search on hyperparameters
`python gridsearch.py`
