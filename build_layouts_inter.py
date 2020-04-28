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

# pipeline:
# apply:
## spatial clustering
## PCA
## entropy
## feature selection
## correlation matrix
# create canvas
# create ruleset

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
df = [d["data"] for d in dts if d["name"] == dataset_name][0].copy(deep=True)

# st.subheader("Columns")
# st.write(np.array(df.columns).reshape(1,-1))
st.subheader("DataFrame")
st.write(df)

st.header("Methods")
st.empty()
st.subheader("Variance Threshold")
var_thres = st.slider("Variance Threshold p * (1 - p) ", 0.1, 0.9, 0.8, 0.05)

variances = lrn.variance_thres(df, var_thres)
if isinstance(variances,np.ndarray):
    st.write("No variance in the threshold")
else:
    st.write(variances)

st.subheader("PCA")
pca_n = st.slider("Componets of PCA ", 1, len(df.columns)//2, 2, 1)

components, variance_ratio = lrn.applyPCA(df, pca_n)
# st.write(components, variance_ratio)

if pca_n == 2:
    d = pd.DataFrame(components, columns=["0", "1"])
    scatter_plot = alt.Chart(d).mark_circle().encode(
        x="0", y="1"
    )
    st.write(scatter_plot)
# else:
#     d = pd.DataFrame(components, columns=[])
#     parallel_coordinates = alt.Chart(d).mark_line(
#         index="count()"
#     ).encode(
#         x="0",
#         y="1",
#         detail="index:O"
#     )
#     st.write(parallel_coordinates)

# st.subheader("Covariance Matrix")
# cov_matrix = df.cov()
# heatmap_cov = sns.heatmap(cov_matrix, annot=True, cmap=plt.cm.Blues)

# st.pyplot()

st.subheader("Correlation Matrix")
corr_matrix = df.corr()
# st.write(cov_matrix)
heatmap_corr = sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Blues)
st.pyplot()

# heatmap = alt.Chart(cov_matrix).mark_rect().encode(
#     x="",
#     y="",
#     z=""
# )

st.header("Spatial Clustering")
# st.write(len(df))

spatial_lbl, spectral_lbl = cacheclusterresults(dataset_name)
df_spatial = pd.DataFrame(spatial_lbl, columns=["labels"])

for labels in [spatial_lbl, spectral_lbl]:
    df_l = pd.DataFrame(labels,columns=["0"])
    histogram = alt.Chart(df_l).mark_rect(

    ).encode(
        x=alt.X("0:O",bin=True),
        y=alt.Y("count()")
    )
    st.write(histogram)