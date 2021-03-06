#!/usr/bin/env python3.9
# *-* coding: utf-8*-*

from pathlib import Path
import numpy as np
from processing import data_processing as dp
from training import classifier as cl


if __name__ == "__main__":
    # Import raw url
    url = 'https://raw.githubusercontent.com/Stradigi-AI/aidev_interview_test/master/data/iris.csv'
    df = dp.read_dataset(url=url)
    # Clean dataset by removing corrupted rows
    df = dp.clean_dataset(df)
    # Assign Features and labels
    x = df.iloc[:, 1:-1].copy()
    y = (df.iloc[:, -1:]).copy()
    labels = np.unique(y.to_numpy()).tolist()
    # Custom preprocessing by normalization
    x = dp.normalization(x)
    y = cl.encoding(y)
    x_train, x_test, y_train, y_test = \
        cl.split_dataset(x, y)
    # Perform trainining
    model = cl.train(x_train, y_train)
    # Perform inference
    cl.test(x_test, y_test, model)
    # Save model
    root = Path(__file__).parent.parent
    name = root / "model/model.pkl"
    cl.save_model(model, labels, str(name))
