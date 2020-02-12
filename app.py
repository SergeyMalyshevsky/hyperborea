import os
import pickle

import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from main import get_column_names, train_model, use_model

app = Flask(__name__)

UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def save_table_dict(table_dict):
    table_file = 'table.sav'
    pickle.dump(table_dict, open(table_file, 'wb'))


def get_table_dict():
    table_file = 'table.sav'
    table_dict = pickle.load(open(table_file, 'rb'))
    return table_dict


@app.route('/')
def main():
    return render_template('./index.html')


@app.route('/about')
def about():
    return render_template('./about.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        dataset = request.files['dataset']
        if dataset:
            filename = secure_filename(dataset.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            dataset.save(file_path)
            columns = get_column_names(file_path)

            table_dict = {}
            table_dict['dataset_file'] = file_path
            save_table_dict(table_dict)

            return render_template('./columns.html', columns=columns)


@app.route('/choose_columns', methods=['GET', 'POST'])
def choose_columns():
    table_dict = get_table_dict()

    if request.method == 'POST':
        label = request.form['label']
        features = request.form.getlist('feature')
        if label and features:
            table_dict['params'] = {
                'label': label,
                'features': features
            }
            save_table_dict(table_dict)

            with open("user_code.txt", "r") as f:
                lines = f.readlines()
            user_code = "".join(lines)

            return render_template('./user_code.html', user_code=user_code)


@app.route('/user_code', methods=['GET', 'POST'])
def user_code():
    if request.method == 'POST':
        user_code = request.form['user_code']
        with open("user_code.txt", 'w') as f:
            f.write(user_code)

    ml_algorithms = {
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'knn': 'Nearest Neighbors',
        # 'rn': 'Radius Neighbors',
        'svm': 'Support Vector Machines',
        'naive_bayes': 'Gaussian Naive Bayes',
        'sgd': 'Stochastic Gradient Descent',
        'mlp': 'Multi-layer Perceptron',
        'rf': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
    }

    return render_template('./train.html', ml_algorithms=ml_algorithms)


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        table_dict = get_table_dict()
        saved_model_filename = "my_model.sav"
        table_dict['saved_model_filename'] = saved_model_filename
        save_table_dict(table_dict)

        score = train_model(table_dict['dataset_file'], table_dict['params'], saved_model_filename, algorithm)
        avg_score = sum(score) / len(score)

        return render_template('./score.html', score=avg_score)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        dataset = request.files['test_data']
        if dataset:
            filename = secure_filename(dataset.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            dataset.save(file_path)

            table_dict = get_table_dict()
            table_dict['test_data_file'] = file_path
            save_table_dict(table_dict)

            columns = get_column_names(file_path)

            return render_template('./predict.html', columns=columns)
        else:
            return '<p>Does not add file</p><a href="/user_code">Return to train model</a>'


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        index = request.form['index']

        table_dict = get_table_dict()
        table_dict['params']['index'] = index

        real_data = pd.read_csv(table_dict['test_data_file'])
        output = use_model(real_data, table_dict['params'], table_dict['saved_model_filename'])

        submission_file = 'submission.csv'
        try:
            os.unlink(os.path.join(app.config['UPLOAD_FOLDER'], submission_file))
        except Exception:
            pass
        output.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], submission_file), index=False)

        return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename=submission_file, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
