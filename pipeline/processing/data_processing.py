import pandas as pd


def read_dataset(url: str) -> pd.DataFrame:
    """Read dataset from raw url link."""
    df = pd.read_csv(url)
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    "Remove empty values."
    nan_value = float("NaN")
    df.replace(r'[^0-9a-zA-Z.]', nan_value, regex=True, inplace=True)
    df.dropna(inplace=True)
    return df


def normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize using Min/Max Normalization."""
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    df=(df-df.min())/(df.max()-df.min())
    return df