# == Imports ==
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb

import math

import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
        'vebosity': 0,
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
    df['Initials'] = df['Name'].replace(title_mapping)

    # -- Age imputation --
    if is_train:
        title_age_median = df.groupdby('Initials')['Age]'].median()
    
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
    



