from copy import deepcopy

import numpy
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_features_and_labels(data, feature_fields, label=None, task_type='classification'):
    y = None

    feature_fields_with_label = deepcopy(feature_fields)
    if label:
        feature_fields_with_label.append(label)
    features = data[feature_fields_with_label]

    if label:
        if task_type == 'classification':
            y = features[label].values
            encoder = LabelEncoder()
            encoder.fit(y)
            print("Encoder classes", encoder.classes_)
            numpy.save('model/classes.npy', encoder.classes_)
        if task_type == 'regression':
            y = features[label].values

    return features, y


def create_preprocessor(X_train):
    # PREPROCESSING

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and
                            X_train[cname].dtype == "object"]

    # Select numeric columns
    numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

    # Bundle preprocessing for numerical and categorical data
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_cols),
            ('cat', categorical_transformer, low_cardinality_cols)
        ])

    return preprocessor


def get_column_names(filename):
    data_columns = pd.read_csv(filename, nrows=0, sep=',').columns.tolist()
    return data_columns
