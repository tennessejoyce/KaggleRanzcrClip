from FeatureExtraction import FeatureExtraction
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score, GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_features(cnn_architecture='resnet18'):
    """Attempt to read features from a file, and if that fails recalculate features."""
    filename = f'saved_features/cnn_pca_{cnn_architecture}.csv'
    try:
        df = pd.read_csv(filename, index_col=0)
    except:
        feature_extraction = FeatureExtraction(batch_size=400, cnn_architecture=cnn_architecture)
        df = feature_extraction.fit_transform()
        df.to_csv(filename)
    return df


def random_forest_pipeline():
    model = Pipeline([('feature_selection', VarianceThreshold()),
                      ('classifier', RandomForestClassifier(n_estimators=64, class_weight='balanced'))])
    param_grid = {'classifier__max_depth': [4, 6, 8],
                  'classifier__min_samples_leaf': [0.01, 0.02, 0.04],
                  'feature_selection__threshold': [1e-2, 1e-3, 1e-4, 1e-5]}
    return model, param_grid


def logistic_sgd_pipeline():
    model = SGDClassifier(loss='log', class_weight='balanced', learning_rate='adaptive',
                          eta0=1, validation_fraction=0.2)
    param_grid = {'alpha': np.geomspace(1e-5, 1, 6),
                  'early_stopping': [True, False],
                  'eta0': [1, 0.1, 0.01]}
    return model, param_grid


# resize_all(data_dir='data', target_reslution=224)

df = extract_features(cnn_architecture='resnet152')
print(df.head())

df_train = df[df.stage == 'train']
df_test = df[df.stage == 'test']

feature_cols = [f'pca_{i}' for i in range(400)]
X = df_train[feature_cols].values

groups = df_train.pop('PatientID').values

y_all = df_train.drop(['StudyInstanceUID', 'stage'] + feature_cols, axis=1).values

num_targets = y_all.shape[1]

# Scale the data so that the dominant PCA feature has unit variance.
X /= np.std(X[:, 0])

scores = []
for i in tqdm(range(num_targets)):
    y = y_all[:, i]
    scoring = make_scorer(roc_auc_score, needs_proba=True)
    model, param_grid = logistic_sgd_pipeline()
    optimizer = GridSearchCV(model, param_grid, scoring=scoring, cv=GroupKFold(), n_jobs=3)
    optimizer.fit(X, y, groups=groups)
    keep_cols = ['param_' + p for p in param_grid.keys()] + ['mean_test_score', 'mean_fit_time']
    cv_results = pd.DataFrame(optimizer.cv_results_)[keep_cols]
    print(cv_results)
    scores.append(optimizer.best_score_)
    print(f'{i} best score: {optimizer.best_score_}')

print(pd.Series(scores))
print(f'Overall: {np.mean(scores)}')
