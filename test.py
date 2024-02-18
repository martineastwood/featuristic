import featurize as ft
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

data = pd.DataFrame(data)
target = pd.Series(target)
data.columns = [str(x) for x in data.columns]

feats = ft.featurize(
    data,
    target,
    problem_type="regression",
    feature_depth=1,
    mrmr_k=100,
    swarm_particles=50,
    swarm_iters=100,
)
