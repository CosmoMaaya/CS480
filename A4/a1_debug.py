# %%

import torchvision
import torch
from torchvision import datasets, transforms
import numpy as np
from multiprocessing import Pool
from scipy.special import logsumexp

MAX_ITER = 10
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
        print(pi)
        for k in range(K_RANGE):
            log_r[:,k] = np.log(pi[k]) - 0.5 * np.sum(np.log(S[k] + EPSILON)) - 0.5 * np.sum((X-mu[k]) ** 2 / (S[k] + EPSILON), axis = 1)
        
        log_r_i = logsumexp(log_r, axis = 1)
        log_r = log_r - log_r_i[:,None]
        loss[iter] = -np.sum(log_r_i)

        if iter > 1 and abs(loss[iter] - loss[iter-1]) <= TOLERANCE * abs(loss[iter]):
            break

        r = np.exp(log_r)
        r_dot_k = np.sum(r, axis = 0)
        pi = r_dot_k / N
        mu = np.matmul(r.T, X) / r_dot_k[:,None]
        S = np.matmul(r.T, X**2) / r_dot_k[:,None] - mu ** 2

    return pi, mu, S, loss

transform = transforms.Compose([transforms.ToTensor()])


# %%
data_train = datasets.MNIST(root = "./data/",
                        transform=transform,
                        train = True,
                        download = True)
idx = data_train.targets == 0
np_X = data_train.data[idx].numpy()
N, d1, d2 = np_X.shape
X = np_X.reshape(N, d1*d2) / 255.0
print(X.dtype)
pi, mu, S, loss = GMM(X, 5)

print(loss)
# GMM(X, 5)

# print(res[4])

# %%



