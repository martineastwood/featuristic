# import featurize as ft
# import pandas as pd
# from sklearn.metrics import mean_absolute_error
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from functools import partial
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LinearRegression


# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

# data = pd.DataFrame(data)
# target = pd.Series(target)
# data.columns = [str(x) for x in data.columns]

# feats = ft.featurize(
#     data,
#     target,
#     problem_type="regression",
#     feature_depth=1,
#     mrmr_k=75,
#     swarm_particles=50,
#     swarm_iters=100,
# )

# f = partial(ft.cost_funcs.regression.linear_reg_mae, X=feats, y=target)

# ga = ft.selection.GeneticAlgorithm(
#     cost_func=f, num_individuals=100, num_features=feats.shape[1], max_iters=100
# )

# cost, genome = ga.optimize()

# feats = feats[feats.columns[genome == 1]]

# print(f"Best cost: {cost}, total features: {genome.sum()}")


# N_SPLITS = 5
# strat_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=8888)
# scores = np.empty(N_SPLITS)

# for idx, (train_idx, test_idx) in enumerate(strat_kf.split(feats, target)):
#     X_train, X_test = feats.iloc[train_idx], feats.iloc[test_idx]
#     y_train, y_test = target[train_idx], target[test_idx]

#     clf = LinearRegression()
#     clf.fit(X_train, y_train)

#     preds = clf.predict(X_test)
#     loss = mean_absolute_error(y_test, preds)
#     scores[idx] = loss

# print(f"mean score: {scores.mean():.5f}")


from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np
import featurize as ft
from functools import partial

# fetch dataset
abalone = fetch_ucirepo(id=1)

# data (as pandas dataframes)
data = abalone.data.features
target = abalone.data.targets["Rings"]

for label in "MFI":
    data[label] = (data["Sex"] == label).astype(int)
del data["Sex"]

f = partial(ft.cost_funcs.classification.knn_accuracy, X=data, y=target)

ga = ft.selection.BinaryGeneticAlgorithm(
    cost_func=f,
    population_size=20,
    num_genes=data.shape[1],
    max_iters=35,
    early_termination_iters=10,
)

cost, genome = ga.optimize()

feats = data[data.columns[genome == 1]]

print(f"Best cost: {cost}, total features: {genome.sum()}")


N_SPLITS = 3
strat_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=8888)
scores = np.empty(N_SPLITS)
for idx, (train_idx, test_idx) in enumerate(strat_kf.split(feats, target)):
    X_train, X_test = feats.iloc[train_idx], feats.iloc[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]

    clf = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="brute")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    loss = accuracy_score(y_test, preds)
    scores[idx] = loss

print(scores.mean())
