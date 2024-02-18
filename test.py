import featurize as ft
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from functools import partial


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

data = pd.DataFrame(data)
target = pd.Series(target)
data.columns = [str(x) for x in data.columns]

feats = ft.featurize(data)

k = int(feats.shape[0] * 0.75)
k = 50
mrmr = ft.selection.MaxRelevanceMinRedundancy(K=k)
feats = mrmr.fit_transform(feats, target)


def cost_function(x0, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X[X.columns[x0 == 1]], y, test_size=0.2
    )
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return mae


f = partial(cost_function, X=feats, y=target)

pso = ft.selection.BinaryParticleSwarmOptimiser(
    num_particles=10, num_dimensions=feats.shape[1]
)

cost, position = pso.optimize(f, max_iter=10)
print(cost, position, position.sum())
