import pandas as pd 

def load_data(filepath, sep='\t'): 
    return pd.read_csv(filepath, sep=sep)

def handle_missing(df): 
    return df.dropna()

def preprocess(filepath): 
    df = load_data(filepath)
    df = handle_missing(df)
    return df