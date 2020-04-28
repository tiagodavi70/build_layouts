import learning as lrn
import pandas as pd
import numpy as np
from preprocess import discretize, normalize, transform_continuous
import os

for path in os.listdir("datasets/"):
    name = path[:-4]
    if os.path.isfile("cluster_results/" + name + "_spectral.npy"):
        print(name + " clusters ok!")
        continue
    elif os.path.isfile("datasets/" + path):
        df = pd.read_csv("datasets/" + path)
        print(name + " size: " + str(len(df)))
        transform_continuous(df)
        
        print("Running spatial cluster for: ", name)
        clusters = lrn.applySpatialClustering(df)
        np.save("cluster_results/" + name + "_spatial", clusters)        
        print("Spatial clustering done")

        print("Running spectral cluster for: ", name)
        clusters = lrn.applySpatialClustering(df)
        np.save("cluster_results/" + name + "_spectral", clusters)        
        print("Spectral clustering done")
    
