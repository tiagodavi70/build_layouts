import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import MinMaxScaler
import numbers


def convert_raw():
    raws = [{
        "name": "abalone",
        "url":"https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/",
        "columns":["Sex", "Length", "Diameter", "Height",
            "Whole weight", "Shucked weight", "Viscera weight",
            "Shell weight", "Rings"],
        "path": "abalone.data"
    }]
    for r in raws:
        if not r["name"]+".csv" in os.listdir("datasets/"):
            print("converting: " + r["name"])
            df = pd.read_csv("datasets/raw/" + r["path"], header=None, names=r["columns"])
            df.to_csv("datasets/" + r["name"]+".csv", index=False)


def checktype(c):
    # subclass usage from: https://stackoverflow.com/questions/934616/how-do-i-find-out-if-a-numpy-array-contains-integers
    # if c.dtype == np.floating or (issubclass(c.dtype.type, numbers.Integral) and len(c.unique()) > 25):
    if c.dtype == np.floating or issubclass(c.dtype.type, numbers.Integral):
        return "cont"
    else:
        return "cat"


def normalize(df, col):
    scaler = MinMaxScaler()
    # transform in column vector and then in row vector for assignment    
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).reshape(1,-1)[0]
    return df


def get_normalized_datasets(include_cat=False):
    paths = [p for p in os.listdir("datasets/") if os.path.isfile("datasets/"+ p)]
    dfs = []
    for p in paths:    
        df = pd.read_csv("datasets/" + p)
        normalizedataset(df, include_cat)
        dfs.append({"name": p[:-4], "data": df})
    return dfs

def normalizedataset(df, include_cat):
    for col in df.columns:
        if checktype(df[col]) == "cont":
            normalize(df, col)
        else:
            if include_cat:
                normalize(discretize(df, col), col)
    return df
    
def discretize(df, col):
    sr = df[col]
    un = sr.unique()
    mapping = {}
    for i in range(len(un)):
        mapping[un[i]] = i
    df[col] = df[col].replace(mapping)
    # st.write(df[col],df)
    return df


def transform_continuous(df):
    for col in df.columns:
        if checktype(df[col]) == "cat":
            normalize(discretize(df, col), col)
    return df

def discretizeall(df):
    for col in df.columns:
        if checktype(df[col]) == "cat":
            discretize(df, col)
    return df

def normalizeall(df):
    for col in df.columns:
        if checktype(df[col]) == "cont":
            normalize(df, col)
    return df

if __name__ == '__main__':
    convert_raw()
    a = get_normalized_datasets()
    a = transform_continuous(a)
    print(a[0]["name"])
        
            
