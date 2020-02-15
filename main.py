import pickle
from copy import deepcopy

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from preprocessing import get_features_and_labels
from user_data_preprocessing import preprocess_data_function


def get_column_names(filename):
    data_columns = pd.read_csv(filename, nrows=0, sep=',').columns.tolist()
    return data_columns


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


def train_model(dataset, fields, saved_model_filename, algorithm='decision_tree', task_type='classification'):
    data = pd.read_csv(dataset)

    # get features and labels
    features, y_train = get_features_and_labels(data, fields['features'], fields['label'], task_type=task_type)

    preprocess_data_func = preprocess_data_function()

    try:
        myMod = compile(preprocess_data_func, '', 'exec')
        exec(myMod, globals())
        features = preprocess_data(features)
    except Exception:
        print("System cannot run user code")

    # features = preprocess_data(features)
    X_train = features[fields['features']]

    # create preprocessor for pipeline
    preprocessor = create_preprocessor(X_train)

    # train model
    if task_type == 'classification':
        if algorithm == 'decision_tree':
            model = DecisionTreeClassifier(random_state=241)
        elif algorithm == 'logistic_regression':
            model = LogisticRegression()
        elif algorithm == 'knn':
            model = KNeighborsClassifier()
        # elif algorithm == 'rn':
        #     model = RadiusNeighborsClassifier()
        elif algorithm == 'svm':
            model = svm.SVC(random_state=241)
        elif algorithm == 'naive_bayes':
            model = GaussianNB()
        elif algorithm == 'sgd':
            model = SGDClassifier(random_state=241)
        elif algorithm == 'mlp':
            model = MLPClassifier(random_state=241)
        elif algorithm == 'rf':
            model = RandomForestClassifier(random_state=241)
        elif algorithm == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=241)

    if task_type == 'regression':
        if algorithm == 'decision_tree':
            model = DecisionTreeRegressor(random_state=241)
        elif algorithm == 'linear_regression':
            model = LinearRegression()
        elif algorithm == 'knn':
            model = KNeighborsRegressor()
        elif algorithm == 'svm':
            model = svm.SVR(random_state=241)
        elif algorithm == 'mlp':
            model = MLPRegressor(random_state=241)
        elif algorithm == 'rf':
            model = RandomForestRegressor(random_state=241)
        elif algorithm == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=241)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)
                               ])

    pipeline.fit(X_train, y_train)

    # save the model to disk
    pickle.dump(pipeline, open(saved_model_filename, 'wb'))

    scoring = None
    if task_type == 'regression':
        scoring = 'neg_mean_absolute_error'
    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
    return score


def use_model(dataset, params, saved_model_filename):
    feature_fields = deepcopy(params['features'])
    feature_fields.append(params['index'])

    real_features, y = get_features_and_labels(dataset, feature_fields)

    preprocess_data_func = preprocess_data_function()

    try:
        myMod = compile(preprocess_data_func, '', 'exec')
        exec(myMod, globals())
        real_features = preprocess_data(real_features)
    except Exception:
        print("System cannot run user code")

    real_X = real_features[params['features']]

    # create preprocessor for pipeline
    preprocessor = create_preprocessor(real_X)

    # load model
    pipeline = pickle.load(open(saved_model_filename, 'rb'))

    predictions = pipeline.predict(real_X)

    # print(real_features[params['index']])

    # print(real_X[params['index']])
    output = pd.DataFrame()
    output = pd.DataFrame({
        params['index']: real_features[params['index']],
        params['label']: predictions
    })

    return output


if __name__ == '__main__':
    # load dataset
    dataset_filename = 'titanic.csv'
    params = {
        'label': 'Survived',
        'features': ["Pclass", "Fare", "Age", "Sex"]
    }
    saved_model_filename = 'model/my_model.sav'

    data_columns = get_column_names(dataset_filename)
    print(data_columns)

    score = train_model(dataset_filename, params, saved_model_filename)
    print("Score: ", sum(score) / len(score))

    # predict by model
    saved_model_filename = 'model/my_model.sav'
    real_data = pd.read_csv('test.csv')
    params = {
        'index': 'PassengerId',
        'label': 'Survived',
        'features': ["Pclass", "Fare", "Age", "Sex"]
    }

    output = use_model(real_data, params, saved_model_filename)
    output.to_csv('submission.csv', index=False)

    # print("Real features: ", real_features)
    print("Prediction:")
    print(output.head())
