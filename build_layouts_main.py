import streamlit as st
import pandas as pd
import numpy as np
import time
from PIL import Image
import altair as alt
import seaborn as sns
from matplotlib import pyplot as plt

import learning as lrn
from preprocess import get_normalized_datasets, transform_continuous


@st.cache
def data(cat):
    return get_normalized_datasets(include_cat=cat)

@st.cache
def cacheclusterresults(name):
    return np.load("cluster_results/" + name + "_spatial.npy"), np.load("cluster_results/" + name + "_spectral.npy")


dts = data(cat=True)

st.sidebar.header("Options")
st.sidebar.text("Number of datasets: " + str(len(dts)))

dataset_name = st.sidebar.selectbox(
    "Select dataset:", 
    [d["name"] for d in dts]
)

st.header("Build Layouts for: " + dataset_name)

@st.cache
def makeallcache(dts, dataset_name):
    df = [d["data"] for d in dts if d["name"] == dataset_name][0].copy(deep=True)
    dict_ = {}
    variances = [{"variances": lrn.variance_thres(df, var_thres), "thres": var_thres} for var_thres in np.arange(.05,.55,.05)]
    dict_["variances"] = [v for v in variances if isinstance(v["variances"], np.ndarray)]
    dict_["components"], dict_["variance_ratio"] = lrn.applyPCA(df, len(df.columns)) 
    dict_["corr_matrix"] = df.corr()
    spatial_lbl, spectral_lbl = cacheclusterresults(dataset_name)
    dict_["df_spatial"] = pd.DataFrame(spatial_lbl, columns=["labels"])
    dict_["df_spectral"] = pd.DataFrame(spectral_lbl, columns=["labels"])
    return dict_


dts = data(True)
data = makeallcache(dts, dataset_name)
var_mean = np.array([data["variances"][i]["variances"] for i in range(len(data["variances"]))]).mean(axis=0).reshape(1, -1)
coor_mean = data["corr_matrix"].mean()
spatial_size, spectral_size = len(data["df_spatial"]["labels"].unique()), len(data["df_spectral"]["labels"].unique())
# choose - attributes | area size | position