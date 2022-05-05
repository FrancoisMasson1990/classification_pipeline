import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def encoding(y):
    # Encode the labels into unique integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(np.ravel(y))
    return y


def split_dataset(x, y):
    # Split the data into test and train
    x_train, x_test, y_train, y_test = \
        train_test_split(x,
                         y,
                         test_size=1/3,
                         random_state=0)
    return x_train, x_test, y_train, y_test

def pipeline(x_train, y_train):
    pipe = Pipeline([
    ('selector', VarianceThreshold()),
    ('classifier', KNeighborsClassifier())
    ])
    pipe.fit(x_train, y_train)
    return pipe


def get_parameters():
    parameters = {
        'selector__threshold': [0, 0.001, 0.01],
        'classifier__n_neighbors': [1, 3, 5, 7, 10],
        'classifier__p': [1, 2],
        'classifier__leaf_size': [1, 5, 10, 15]
    }
    return parameters

def grid(pipe, parameters, x_train, y_train):
    grid = GridSearchCV(pipe,
                        parameters,
                        cv=2).fit(x_train, y_train)
    return grid


def train(x, y):
    y = encoding(y)
    x_train, x_test, y_train, y_test = split_dataset(x, y)
    pipe = pipeline(x_train, y_train)
    parameters = get_parameters()
    results = grid(pipe, parameters, x_train, y_train)
    # Access the best set of parameters
    best_params = results.best_params_
    best_pipe = results.best_estimator_
    

    # print('Training set score: ' + str(pipe.score(X_train,y_train)))
    # print('Test set score: ' + str(pipe.score(X_test,y_test)))

    # print('Training set score: ' + str(grid.score(X_train, y_train)))
    # print('Test set score: ' + str(grid.score(X_test, y_test)))

    # # Access the best set of parameters
    # best_params = grid.best_params_
    # print(best_params)
    # # Stores the optimum model in best_pipe
    # best_pipe = grid.best_estimator_
    # print(best_pipe)
    # import pandas as pd
    # result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
    # print(result_df.columns)