import pandas as pd


def read_dataset(url):
    df = pd.read_csv(url)
    return df
