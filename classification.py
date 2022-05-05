#!/usr/bin/env python3.9
# *-* coding: utf-8*-*

from pipeline.processing import data_processing as dp
from pipeline.training import classifier as cl


if __name__ == "__main__":
    # Import raw url
    url = 'https://raw.githubusercontent.com/Stradigi-AI/aidev_interview_test/master/data/iris.csv'
    df = dp.read_dataset(url=url)
    df = dp.clean_dataset(df)
    # The data matrix X
    x = df.iloc[:,1:-1].copy()
    # The labels
    y = (df.iloc[:,-1:]).copy()
    x = dp.normalization(x)
    cl.train(x, y)

