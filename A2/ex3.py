import sys
import numpy as np

#Exercise 3
#Usage: python3 ex3.py X_train Y_train X_test Y_test C eps

def SVR(X_train, Y_train, C, eps):
    max_pass, eta = 500, 0.001
    n, d = X_train.shape
    w, b = np.zeros(d), 0
    #Implement me! You may choose other parameters eta, max_pass, etc. internally
    for t in range(max_pass):
        # w_new, b_new = np.zeros(d), 0
        for i in range(n):
            # print(len(w))
            diff = Y_train[i] - (np.dot((w.T), X_train[i])  + b)
            # print(diff)
            if diff >= eps:
                w = w + X_train[i] * C * eta
                b = b + C * eta
            elif diff <= -eps:
                w = w - X_train[i] * C * eta
                b = b - C * eta
            w = w / (1+eta)
    #Return: parameter vector w, b
    return w, b

def compute_loss(X, Y, w, b, C, eps):
    #Implement meCC
    #Return: loss computed on the given set
    error = compute_error(X, Y, w, b, C, eps)
    return error + np.linalg.norm(w) ** 2 / 2.0

def compute_error(X, Y, w, b, C, eps):
    #Implement me!
    #Return: error computed on the given set
    n, d = X.shape
    error = 0
    for i in range(n):
        diff = np.absolute(Y[i] - (X[i].dot(w) + b)) - eps
        if diff > 0:
            error += diff
    return error * C

if __name__ == "__main__":
    args = sys.argv[1:]
    #You may import the data some other way if you prefer
    X_train = np.loadtxt(args[0], delimiter=",")
    Y_train = np.loadtxt(args[1], delimiter=",")
    X_test = np.loadtxt(args[2], delimiter=",")
    Y_test = np.loadtxt(args[3], delimiter=",")
    C = float(args[4])
    eps = float(args[5])
    
    w, b = SVR(X_train, Y_train, C, eps)
    print(f"training error: {compute_error(X_train, Y_train, w, b, C, eps)}")
    print(f"training loss: {compute_loss(X_train, Y_train, w, b, C, eps)}")
    print(f"test error: {compute_error(X_test, Y_test, w, b, C, eps)}")