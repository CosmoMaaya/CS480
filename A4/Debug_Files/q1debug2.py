# %%
import torchvision
import torch
from torchvision import datasets, transforms
import numpy as np
# from tqdm import tqdm
from scipy.special import logsumexp
from multiprocessing import Pool

MAX_ITER = 300
EPSILON = 1e-7
TOLERANCE = 1e-5

# %%
def initializeModel(K, d):
    np.random.seed(0)
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)

    np.random.seed(0)
    mu = np.random.normal(0, 3, size=(K, d))

    np.random.seed(0)
    S = np.random.rand(K, d) + 0.5 + EPSILON

    return pi, mu, S

# %%
def GMM(X, K_RANGE):
    N, d = X.shape
    pi, mu, S = initializeModel(K_RANGE, d)
    log_r = np.zeros((N, K_RANGE))
    loss = [0.0] * MAX_ITER

    # for iter in tqdm(range(MAX_ITER), total=MAX_ITER):
    for iter in range(MAX_ITER):
        # print(f"{iter}th pi: {pi}")
        # print(f"{iter}th S: {S}")
        # print(f"{iter}th {np.log(pi[3])}")
        for k in range(K_RANGE):
            # exp_power = np.dot((X-mu[k]) ** 2, 1/S[k]) * (-1/2)
            # if iter==2:
                # print(pi[k] * np.power(np.prod(S[k]), -1/2) * np.exp(exp_power))
            # r[:,k] = pi[k] * np.power(np.prod(S[k]), -1/2) * np.exp(exp_power)
            log_r[:,k] = np.log(pi[k]) - 0.5 * np.sum(np.log(S[k] + EPSILON)) - 0.5 * np.dot((X-mu[k]) ** 2, 1/(S[k] + EPSILON))
        r_total = logsumexp(log_r, axis=1)
        log_r = log_r - r_total[:,None]
        loss[iter] = -np.sum(r_total)

        if iter > 1 and abs(loss[iter] - loss[iter-1]) <= TOLERANCE * abs(loss[iter]):
            break
        
        r = np.exp(log_r)
        r_dot_k = np.exp(logsumexp(log_r, axis=0))
        # print(r_dot_k)
        pi = r_dot_k / N
        mu = np.dot(r.T, X) / r_dot_k[:,None]
        S = np.dot(r.T, X ** 2) / r_dot_k[:,None] - mu ** 2 

    return pi, mu, S, loss

# %%
transform = transforms.Compose([transforms.ToTensor()])
data_train = datasets.MNIST(root = "./data/",
                        transform=transform,
                        train = True,
                        download = True)
idx = data_train.targets == 0
np_X = data_train.data[idx].numpy()
N, d1, d2 = np_X.shape
X = np_X.reshape(N, d1*d2)
X = X.astype('int64') / 255
pi, mu, S, loss = GMM(X, 5)
print(pi)
print(mu)
print(S)
print(loss)