# Normalization and standardization logic goes here

import pickle
from sklearn.preprocessing import MinMaxScaler


with open("data/min_max_scaling.pkl", "rb") as pickle_file:
    min_max_scaling :MinMaxScaler = pickle.load(pickle_file)


def perform_min_max_scaling(data):
    data = min_max_scaling.transform(data)        
    return data