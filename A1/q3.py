import numpy as np
from matplotlib import pyplot as plt
from time import process_time_ns

from sklearn.linear_model import LinearRegression, Ridge, Lasso

def split(data_X, k):
    N = len(data_X)
    M = N // k
    partitions = [None for _ in range(k)]
    for i in range(k):
        test_min = i * M
        test_max = (i+1) * M - 1
        test_max = N-1 if test_max >= N else test_max
        test = [i for i in range(test_min, test_max+1)]
        train = [i for i in range(N) if i not in test]
        partitions[i] = (train, test)
    return partitions

for dataset in ['A', 'B', 'C']:
    k = 10
    data_X = np.loadtxt(open(f"data/X_train_{dataset}.csv", "rb"), delimiter=",")
    data_y = np.loadtxt(open(f"data/Y_train_{dataset}.csv", "rb"), delimiter=",")
    test_data_X = np.loadtxt(open(f"data/X_test_{dataset}.csv", "rb"), delimiter=",")
    test_data_y = np.loadtxt(open(f"data/Y_test_{dataset}.csv", "rb"), delimiter=",")
    partitions = split(data_X, k)

    hypers = [i for i in range(1, 11)]
    hypers = [0.1] + hypers
    performances = [[0] * len(hypers) for _ in range(2)]
    for hyper_idx, hyper in enumerate(hypers):
        # enumerate splits
        for train, test in split(data_X, k):
            # print(train, test)
            models = [Ridge(alpha=hyper), Lasso(alpha=hyper)]
            for i, model in enumerate(models):
                model.fit(data_X[train], data_y[train])
                predict = model.predict(data_X[test])
                mse = np.square(np.subtract(predict, data_y[test])).mean()
                performances[i][hyper_idx] += mse
    best_lambdas = []
    for idx in range(2):
        best_lambdas.append(hypers[performances[idx].index(min(performances[idx]))])
    models = [LinearRegression(), Ridge(alpha=best_lambdas[0]), Lasso(alpha=best_lambdas[1])]
    for idx, model in enumerate(models):
        model_name = type(model).__name__
        model.fit(data_X, data_y)
        w = model.coef_
        predicated_values = model.predict(test_data_X)
        mse = np.square(np.subtract(predicated_values, test_data_y)).mean()
        if model_name=="LinearRegression":
            print(f"Model: {type(model).__name__} gives mse = {mse}")
        else:
            print(f"Model: {type(model).__name__} gives mse = {mse} with lambda = {best_lambdas[idx-1]}")
             
        plt.hist(w, bins=20, label=model_name, alpha=0.3)
        plt.legend(loc='best')
    plt.title(f"Histogram of Parameter Value Frequency for dataset {dataset}")
    plt.xlabel("Pass Number")
    plt.ylabel("Training Loss")
    plt.savefig(f'Exercise3_dataset_{dataset}.png')
    plt.close()
    # plt.show()