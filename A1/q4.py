import numpy as np
from matplotlib import pyplot as plt

def partition(list, left, right):
    x = list[right]
    i = left
    for j in range(left, right):
        if list[j] <= x:
            list[i], list[j] = list[j], list[i]
            i += 1
    list[i], list[right] = list[right], list[i]
    return i

def quickSelect(list, left, right, k):
    if (k > 0 and k <= right - left + 1):
        index = partition(list, left, right)
        if (index - left == k - 1):
            return list[index]
        
        if (index - left > k - 1):
            return quickSelect(list, left, index - 1, k)
        
        return quickSelect(list, index + 1, right, k - index + left - 1)


def k_nearest_regression(X_train, y_train, X, k):
    dists = [np.linalg.norm((x_i - X)) for x_i in X_train]
    kth = quickSelect(dists[:], 0, len(dists) - 1, k)
    sum = 0
    count = 0
    for i, distance in enumerate(dists):
        if distance <= kth:
            sum += y_train[i]
            count += 1
            if count == k:
                break
    return sum / k

def least_sqaure(X, y, n):
    A = np.c_[X, np.ones(n)]
    z = np.r_[y]

    LEFT = np.dot(A.T, A)
    RIGHT = np.dot(A.T, z)
    wb = np.linalg.solve(LEFT, RIGHT)
    return wb
# Debug:

# data_X = np.loadtxt(open(f"data/X_train_F.csv", "rb"), delimiter=",")
# data_y = np.loadtxt(open(f"data/Y_train_F.csv", "rb"), delimiter=",")
# test_data_X = np.loadtxt(open(f"data/X_test_F.csv", "rb"), delimiter=",")
# test_data_y = np.loadtxt(open(f"data/Y_test_F.csv", "rb"), delimiter=",")
# kNN_predicts = [None for _ in range(9)]
# predicts = [k_nearest_regression(data_X, data_y, X, 9) for X in test_data_X]

# Question2
datasets = ['D', 'E']
for dataset in datasets:
    data_X = np.loadtxt(open(f"data/X_train_{dataset}.csv", "rb"), delimiter=",")
    data_y = np.loadtxt(open(f"data/Y_train_{dataset}.csv", "rb"), delimiter=",")
    test_data_X = np.loadtxt(open(f"data/X_test_{dataset}.csv", "rb"), delimiter=",")
    test_data_y = np.loadtxt(open(f"data/Y_test_{dataset}.csv", "rb"), delimiter=",")
    n = data_X.shape[0]
    wb = least_sqaure(data_X, data_y, n)

    #predict on every value
    x_min = min(test_data_X)
    x_max = max(test_data_X)
    x_values = np.arange(x_min, x_max, 0.001)

    linear_predicts_plot = np.dot(np.c_[x_values, np.ones(len(x_values))], wb)
    k1_predicts = [k_nearest_regression(data_X, data_y, X, 1) for X in x_values]
    k9_predicts = [k_nearest_regression(data_X, data_y, X, 9) for X in x_values]

    plt.xlabel("X")
    plt.ylabel("y")
    plt.plot(x_values, linear_predicts_plot, label="Least Squares Linear Regression")
    plt.plot(x_values, k1_predicts, label="1-nearest Neighbour Regression")
    plt.plot(x_values, k9_predicts, label="9-nearest Neighbour Regression")
    plt.legend(loc='best')
    plt.title(f"Predicted y Values for Set x of Dataset {dataset}")
    plt.savefig(f'Exercise4_dataset_{dataset}_all_points.png')
    # plt.show()
    plt.close()

    linear_predicts = np.dot(np.c_[test_data_X, np.ones(n)], wb)

    kNN_predicts = [None for _ in range(9)]
    for k in range(1, 10):
        predicts = [k_nearest_regression(data_X, data_y, X, k) for X in test_data_X]
        kNN_predicts[k-1] = predicts
    mses = [0 for _ in range(9)]
    for i in range(9):
        mses[i] = np.square(np.subtract(kNN_predicts[i], test_data_y)).mean()
        # print(np.subtract(kNN_predicts[i], test_data_y))

    ks = [i + 1 for i in range(9)]
    linear_error = np.square(np.subtract(linear_predicts, test_data_y)).mean()
    plt.xlabel("k")
    plt.ylabel("Mean-Squared Error")
    plt.plot(ks, mses, label="kNN-Error")
    plt.plot(ks, [linear_error] * len(ks), label="Linear Regression Error")
    plt.legend(loc='best')
    plt.title(f"Mean-Squared Erros of Dataset {dataset}")
    plt.savefig(f'Exercise4_dataset_{dataset}_error.png')
    # plt.show()
    plt.close()

    plt.scatter(data_X, data_y, label="train")
    plt.scatter(test_data_X, test_data_y, label="test")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Data points for {dataset}")
    plt.legend(loc='best')
    plt.savefig(f'Exercise4_dataset_{dataset}_datapoints.png')
    plt.close()
    # plt.show()

# Question3

data_X = np.loadtxt(open(f"data/X_train_F.csv", "rb"), delimiter=",")
data_y = np.loadtxt(open(f"data/Y_train_F.csv", "rb"), delimiter=",")
test_data_X = np.loadtxt(open(f"data/X_test_F.csv", "rb"), delimiter=",")
test_data_y = np.loadtxt(open(f"data/Y_test_F.csv", "rb"), delimiter=",")
n, d = data_X.shape
wb = least_sqaure(data_X, data_y, n)
linear_predicts = np.dot(np.c_[test_data_X, np.ones(n)], wb)

kNN_predicts = [None for _ in range(9)]
for k in range(1, 10):
    predicts = [k_nearest_regression(data_X, data_y, X, k) for X in test_data_X]
    kNN_predicts[k-1] = predicts

mses = [0 for _ in range(9)]
for i in range(9):
    mses[i] = np.square(np.subtract(kNN_predicts[i], test_data_y)).mean()

ks = [i + 1 for i in range(9)]
linear_error = np.square(np.subtract(linear_predicts, test_data_y)).mean()
plt.xlabel("k")
plt.ylabel("Mean-Squared Error")
plt.plot(ks, mses, label="kNN-Error")
plt.plot(ks, [linear_error] * len(ks), label="Linear Regression Error")
plt.legend(loc='best')
plt.title("Mean-Squared Errors of Dataset F")
plt.savefig(f'Exercise4_dataset_F_error.png')
# plt.show()
plt.close()
