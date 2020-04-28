from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering

import streamlit as st


def variance_thres(X, thres):
    selector = VarianceThreshold(threshold=(thres * (1 - thres)))
    try:
        selector.fit(X)
        return selector.variances_
    except ValueError:
        return None


def applyPCA(X, n):
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca.transform(X), pca.explained_variance_ratio_


def applySpatialClustering(X):
    clustering = MeanShift().fit(X)
    return clustering.labels_

def applySpectralClustering(X):
    clustering = SpectralClustering().fit(X)
    return clustering.labels_