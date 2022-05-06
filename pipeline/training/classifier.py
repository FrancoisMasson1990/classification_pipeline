#!/usr/bin/env python3.9
# *-* coding: utf-8*-*

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os


def encoding(y: pd.DataFrame) -> np.ndarray:
    """Encode the labels into unique integers."""
    encoder = LabelEncoder()
    y = encoder.fit_transform(np.ravel(y))
    return y


def split_dataset(x: pd.DataFrame, y: pd.DataFrame) -> List[pd.DataFrame]:
    """Split the data into test and train."""
    x_train, x_test, y_train, y_test = \
        train_test_split(x,
                         y,
                         test_size=1/3,
                         random_state=0)
    return [x_train, x_test, y_train, y_test]


def pipeline(x_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """Generate the pipeline."""
    pipe = Pipeline([
        ('selector', VarianceThreshold()),
        ('classifier', KNeighborsClassifier())
        ])
    pipe.fit(x_train, y_train)
    return pipe


def get_parameters() -> Dict:
    """Define parameters for gridsearch."""
    parameters = {
        'selector__threshold': [0, 0.001, 0.01],
        'classifier__n_neighbors': [1, 3, 5, 7, 10],
        'classifier__p': [1, 2],
        'classifier__leaf_size': [1, 5, 10, 15]
    }
    return parameters


def grid(pipe: Pipeline,
         parameters: Dict,
         x_train: pd.DataFrame,
         y_train: pd.DataFrame) -> GridSearchCV:
    """Apply Grid Search method."""
    grid = GridSearchCV(pipe,
                        parameters,
                        cv=2).fit(x_train, y_train)
    return grid


def train(x_train: pd.DataFrame,
          y_train: pd.DataFrame) -> GridSearchCV:
    """Perform training."""
    pipe = pipeline(x_train, y_train)
    parameters = get_parameters()
    model = grid(pipe, parameters, x_train, y_train)
    # Access the best set of parameters
    best_params = model.best_params_
    best_pipe = model.best_estimator_
    print('Training set score: ' + str(model.score(x_train,
                                                   y_train)))
    return model


def test(x_test: pd.DataFrame,
         y_test: pd.DataFrame,
         model: GridSearchCV) -> None:
    """Perform inference."""
    print('Test set score: ' + str(model.score(x_test, y_test)))


def save_model(model: GridSearchCV, name: str = "model.pkl") -> None:
    if not os.path.exists(name):
        folder = ('/').join(name.split('/')[:-1])
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, name)


def load_model(name: str) -> Optional[GridSearchCV]:
    model = joblib.load(name)
    return model
