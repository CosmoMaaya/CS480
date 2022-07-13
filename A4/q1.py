
import torchvision
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

MAX_ITER = 100
EPSILON = 1e-10
TOLERANCE = 1e-5

def initializeModel(K, d):
    np.random.seed(0)
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)

    np.random.seed(0)
    mu = np.random.normal(0, 3, size=(K, d))

    np.random.seed(0)
    S = np.random.rand(K, d) + 0.5

    return pi, mu, S

def GMM(X, K_RANGE):
    N, d = X.shape
    pi, mu, S = initializeModel(K_RANGE, d)
    log_r = np.zeros((N, K_RANGE))
    loss = [0.0] * MAX_ITER

    for iter in range(MAX_ITER):
        for k in range(K_RANGE):
            for i in range(N):
                log_prob = np.log(pi[k])
                for j in range(d):
                    # print(k, i, j)
                    if np.isclose(S[k,j], 0):
                        if  np.isclose(X[i,j], mu[k,j]):
                            mus = np.argwhere(mu[:,j]== mu[k,j]).flatten()
                            sigmas = np.argwhere(S[:,j] == 0).flatten()
                            num_models = len(np.intersect1d(mus, sigmas))
                            log_prob += np.log(1/num_models)
                        else:
                            log_prob = float('-inf')
                            break
                    else:
                        # if iter == 2 and k==5:
                            # print(i, k, j, S[k,j])
                        log_prob += -0.5 * np.log(S[k][j]) - 0.5 * (X[i][j] - mu[k][j]) ** 2 / S[k][j]
                log_r[i,k] = log_prob
                # log_r[:,k] = np.log(pi[k]) - 0.5 * np.sum(np.log(S[k] + EPSILON)) - 0.5 * np.sum((X-mu[k]) ** 2 / (S[k] + EPSILON), axis = 1)
        log_r_i = logsumexp(log_r, axis = 1)
        log_r = log_r - log_r_i[:,None]
        loss[iter] = -np.sum(log_r_i)
        print(loss[iter])
        if iter > 1 and abs(loss[iter] - loss[iter-1]) <= TOLERANCE * abs(loss[iter]):
            break

        r = np.exp(log_r)
        r_dot_k = np.sum(r, axis = 0)
        pi = r_dot_k / N
        mu = np.matmul(r.T, X) / r_dot_k[:,None]
        S = np.matmul(r.T, X**2) / r_dot_k[:,None] - mu ** 2

    return pi, mu, S, loss

transform = transforms.Compose([transforms.ToTensor()])


X = np.loadtxt(open("./A4/gmm_dataset.csv", "rb"), delimiter=",")
k = 5
pi, mu, S, loss = GMM(X, k+1)

