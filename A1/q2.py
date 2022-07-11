from re import L
import numpy as np
from matplotlib import pyplot as plt
from time import process_time
from sklearn import linear_model

def cal_error(X, y, w, b):
    n, d = X.shape
    return (np.linalg.norm(np.dot(X, w) + np.ones(n) * b - y) ** 2) / (2*n)

def cal_error_loss(X, y, w, b, lambda_):
    error = cal_error(X, y, w, b)
    return error, error + lambda_ * (np.linalg.norm(w) ** 2)

def closedFormRidge(X, y, lambda_):
    n, d = X.shape
    A_1 = np.c_[X, np.ones(n)]
    A_2 = np.c_[np.identity(d) * np.sqrt(2.0*lambda_*n), np.zeros(d)]
    A = np.r_[A_1, A_2]
    # for i in range(n, n+d):
        # print(A[i])
    z = np.r_[y, np.zeros(d)]

    LEFT = np.dot(A.T, A)
    RIGHT = np.dot(A.T, z)
    wb = np.linalg.solve(LEFT, RIGHT)
    return wb[:-1], wb[-1]

def gradientRidge(X, y, lambda_, max_pass):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    step_size = 0.001
    tol = 0.00001
    mistake = [0] * max_pass 
    for t in range(max_pass):
        # w_grad = 1.0/n * np.dot(X.T, (np.dot(X, w) + b - y)) + 2 * lambda_ * w
        # b_grad = 1.0/n * np.dot(np.ones(n).T, (np.dot(X, w) + b - y))
        bracket = np.dot(X, w) + np.ones(n) * b - y
        wt = w - (np.dot(X.T, bracket) / n + 2 * lambda_ * w) * step_size
        bt = b - (np.dot(np.ones(n), bracket) / n) * step_size
        # wt = w - step_size * w_grad
        # bt = b - step_size * b_grad
        if np.linalg.norm(wt - w) <= tol:
            break
        w = wt
        b = bt
        _, train_loss = cal_error_loss(X, y, w, b, lambda_)
        mistake[t] = train_loss
    return w, b, mistake

max_pass = 2500
X_test = np.loadtxt(open("data/housing_X_test.csv", "rb"), delimiter=",").transpose()
y_test = np.loadtxt(open("data/housing_y_test.csv", "rb"), delimiter=",")
X_train = np.loadtxt(open("data/housing_X_train.csv", "rb"), delimiter=",").transpose()
y_train = np.loadtxt(open("data/housing_y_train.csv", "rb"), delimiter=",")


# standardize
X_test_std = (X_test - np.mean(X_train)) / np.std(X_train)
X_train_std = (X_train - np.mean(X_train)) / np.std(X_train)

t1_start = process_time()
w_minimum_0, b_minimum_0 = closedFormRidge(X_train_std, y_train, 0)
t1_stop = process_time()
min_0_train_err, min_0_train_loss = cal_error_loss(X_train_std, y_train, w_minimum_0, b_minimum_0, 0)
min_0_test_err = cal_error(X_test_std, y_test, w_minimum_0, b_minimum_0)
print("With lambda 0, linear regression gives: train_error: ", min_0_train_err, "train_loss: ", min_0_train_loss, "test_error: ", min_0_test_err, "CPU Time: ", t1_stop-t1_start)
# print(w_minimum_0, b_minimum_0)
t1_start = process_time()
w_minimum_10, b_minimum_10 = closedFormRidge(X_train_std, y_train, 10)
t1_stop = process_time()
min_10_train_err, min_10_train_loss = cal_error_loss(X_train_std, y_train, w_minimum_10, b_minimum_10, 10)
min_10_test_err = cal_error(X_test_std, y_test, w_minimum_10, b_minimum_10)
print("With lambda 10, linear regression gives: train_error: ", min_10_train_err, "train_loss: ", min_10_train_loss, "test_error: ", min_10_test_err, "CPU Time: ", t1_stop-t1_start)
# print(w_minimum_10, b_minimum_10)
# Gradient Descent After std
t1_start = process_time()
w_grd_0, b_grd_0, grd_0_train_loss = gradientRidge(X_train_std, y_train, 0, max_pass)
t1_stop = process_time()
grd_0_train_err = cal_error(X_train_std, y_train, w_grd_0, b_grd_0)
grd_0_test_err = cal_error(X_test_std, y_test, w_grd_0, b_grd_0)
print("With lambda 0, gradient descent gives: train_error: ", grd_0_train_err, "test_error: ", grd_0_test_err, "CPU Time: ", t1_stop-t1_start)
# print(w_grd_0, b_grd_0)
t1_start = process_time()
w_grd_10, b_grd_10, grd_10_train_loss = gradientRidge(X_train_std, y_train, 10, max_pass)
t1_stop = process_time()
grd_10_train_err = cal_error(X_train_std, y_train, w_grd_10, b_grd_10)
grd_10_test_err = cal_error(X_test_std, y_test, w_grd_10, b_grd_10)
print("With lambda 10, gradient descent gives: train_error: ", grd_10_train_err, "test_error: ", grd_10_test_err, "CPU Time: ", t1_stop-t1_start)
# print(w_grd_10, b_grd_10)
plt.xlabel("Pass Number")
plt.ylabel("Training Loss")
plt.plot(range(max_pass), grd_0_train_loss, label="Lambda=0")
plt.plot(range(max_pass), grd_10_train_loss, label="Lambda=10")
plt.legend(loc='best')
plt.title("Training Loss over Iterations")
plt.savefig("Exercise2.png")
# plt.show()
plt.close()
