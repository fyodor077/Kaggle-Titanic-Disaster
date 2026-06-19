# == Imports ==
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

import math

import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline

# == Config ==
CONFIG = {
    # Paths
    'train_path': 'data/train.csv',
    'test_path': 'data/test.csv',

    # Target
    'target': 'Survived',

    # Cross-validation (CV)
    'n_folds': 5,
    'seed': 1337,

    # Features
    'cat_features': ['Embarked', 'Initials'],
    'drop_features': ['Name', 'Ticket', 'Cabin', 'PassengerId'],

    # Model params
    'xgb_params': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 1337,
        'verbosity': 0,
    },
    'lgb_params': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 1337,
        'verbosity': -1,
    }
}
# == Load data ==
def load_data(config):
    train = pd.read_csv(config['train_path'])
    test = pd.read_csv(config['test_path'])
    test_ids = test['PassengerId'].copy()
    return train, test, test_ids

train, test, test_ids = load_data(CONFIG)

# == Feature Engineering ==
def engineer_features(df, is_train=True, title_age_median=None, fare_bin_edges=None, fare_median_by_class=None):
    df = df.copy()

    # -- Initials --
    title_mapping = {
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'MMe': 'Mrs',
    'Dona': 'Mrs',
    'Lady': 'Mrs',
    'Countess': 'Mrs',
    'Capt':'Officer',
    'Col':'Officer',
    'Major':'Officer',
    'Dr':'Officer',
    'Rev':'Officer',
    'Sir': 'Mr',
    'Don': 'Mr',
    'Jonkheer': 'Mr'
    }
    df['Initials'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Initials'] = df['Initials'].replace(title_mapping)

    # -- Age imputation --
    if is_train:
        title_age_median = df.groupby('Initials')['Age'].median()
    
    overall_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(df['Initials'].map(title_age_median))
    df['Age'] = df['Age'].fillna(overall_median)

    # -- AgeBin --
    age_bins = [-0.01, 12, 60, np.inf]
    age_labels = [0, 1, 2]
    df['AgeBin'] = pd.cut(df['Age'], bins = age_bins, labels = age_labels).astype(int)

    # -- Sex --
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # -- FamilySize --
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # -- FamilySizeBin --
    family_bins = [0, 1, 4, np.inf]
    family_labels = [0, 1, 2] # (0, 1] = (0) Alone, (1, 4] = (1) Small Family, (4, np.inf) = (2) Large Family
    df['FamilySizeBin'] = pd.cut(df['FamilySize'], bins=family_bins, labels=family_labels).astype(int)

    # -- Fare imputation (for TEST only) --
    if is_train:
        fare_median_by_class = df.groupby('Pclass')['Fare'].median()
    
    df['Fare'] = df['Fare'].fillna(df['Pclass'].map(fare_median_by_class))

    # -- FareBin --
    if is_train:
        df['FareBin'], fare_bin_edges = pd.qcut(df['Fare'], q=5, retbins=True, labels=[0,1,2,3,4])
        fare_bin_edges[0] = -np.inf
        fare_bin_edges[-1] = np.inf
    else:
        df['FareBin'] = pd.cut(df['Fare'], bins=fare_bin_edges, labels=[0,1,2,3,4])

    # -- HasCabin --
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # -- Embarked imputation --
    df['Embarked'] = df['Embarked'].fillna(train['Embarked'].mode()[0])

    # -- Drop --
    df = df.drop(columns=CONFIG['drop_features'])

    return df, title_age_median, fare_bin_edges, fare_median_by_class

# == Apllying ==
train, title_age_median, fare_bin_edges, fare_median_by_class = engineer_features(
    train, is_train=True)
test, _, _, _ = engineer_features(
    test,
    is_train=False,
    title_age_median=title_age_median,
    fare_bin_edges=fare_bin_edges,
    fare_median_by_class=fare_median_by_class
)

# == Preprocessing ==
def build_preprocessor(cat_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(
                handle_unknown='ignore', # в test игнорируем категорию, которой не было в train (0)
                sparse_output=False
            ), cat_features),
        ],
        remainder='passthrough' # все остальные фичи (не категориальные) не трогаем
    )
    return preprocessor

# -- Target / Features split --
X_train = train.drop(columns=[CONFIG['target']])
y_train = train[CONFIG['target']]
X_test = test.copy()

preprocessor = build_preprocessor(CONFIG['cat_features'])

# == Cross-Validation & Training ==
def train_model(model, X_train, y_train, X_test, preprocessor, config):
    """
    Обучаем модель StratifiedKFold Cross-Validation.
    Возвращаем out-of-fold preds, test preds, scores.

    """
    skf = StratifiedKFold(
        n_splits=config['n_folds'],
        shuffle=True,
        random_state=config['seed']
    )

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'Fold {fold + 1}/{config["n_folds"]}', end=' | ')
        
        # -- Split --
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # -- Preprocessing --
        # fit только на трейн фолде
        fold_preprocessor = clone(preprocessor)
        X_fold_train = fold_preprocessor.fit_transform(X_fold_train)
        X_fold_val = fold_preprocessor.transform(X_fold_val)
        X_fold_test = fold_preprocessor.transform(X_test)

        # -- Training --
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)

        # -- Validation --
        val_preds = fold_model.predict(X_fold_val)
        fold_score = accuracy_score(y_fold_val, val_preds)
        scores.append(fold_score)
        print(f'Accuracy: {fold_score:.4f}')

        # -- Out-of-Fold & Test predictions --
        oof_preds[val_idx] = val_preds
        test_preds += fold_model.predict(X_fold_test) / config['n_folds']
    
    # -- Cross-Validation Summary --
    print(f'\nMean Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'Out-Of-Fold Accuracy: {accuracy_score(y_train, oof_preds):.4f}')

    return oof_preds, test_preds, scores

# == Models ==
baseline = LogisticRegression(
    max_iter=1000,
    random_state=CONFIG['seed']
)
xgb_model = xgb.XGBClassifier(**CONFIG['xgb_params'])
lgb_model = lgb.LGBMClassifier(**CONFIG['lgb_params'])

models = {
    'Baseline (LogReg)': baseline,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
}

# == Run CV & Training ==
results = {}

for name, model in models.items():
    print(f'\n{"="*40}')
    print(f'Model: {name}')
    print(f'{"="*40}')

    oof_preds, test_preds, scores = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        preprocessor=preprocessor,
        config=CONFIG
    )
    results[name] = {
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
    }

# == Restults Summary ==
print(f'\n{"="*40}')
print('Results Summary')
print(f'{"="*40}')
for name, result in results.items():
    print(f'{name:25s} | Accuracy: {result["mean_score"]:.4f} ± {result["std_score"]:.4f}')

# == Submission ==
def make_submission(results, test_ids, config):
    """
    Выбираем лучшую модель по 'mean CV score',
    округляем предсказания и сохраняем их в submission.csv.
    """
    best_model_name = max(results, key=lambda x: results[x]['mean_score'])
    best_test_preds = results[best_model_name]['test_preds']

    print(f'\nBest model: {best_model_name}')
    print(f'Best CV Accuracy: {results[best_model_name]["mean_score"]:.4f}')

    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': np.round(best_test_preds).astype(int)
    })

    submission.to_csv('submission.csv', index=False)
    print(f'\nSubmission saved: submission.csv')
    print(submission['Survived'].value_counts())

make_submission(results, test_ids, CONFIG)
    
