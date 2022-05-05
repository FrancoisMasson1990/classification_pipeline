#!/usr/bin/env python3.9
# *-* coding: utf-8*-*

from pipeline.processing import data_processing as dp


if __name__ == "__main__":
    # Import
    url = 'https://raw.githubusercontent.com/Stradigi-AI/aidev_interview_test/master/data/iris.csv'
    df = dp.read_dataset(url=url)
    print(df)