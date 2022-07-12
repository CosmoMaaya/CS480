
import torchvision
import torch
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

MAX_ITER = 50
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
    r = np.zeros((N, K_RANGE))
    loss = [0.0] * MAX_ITER

    for iter in tqdm(range(MAX_ITER), total=MAX_ITER):
        for k in range(K_RANGE):
            exp_power = np.dot((X-mu[k]) ** 2, 1/S[k]) * (-1/2)
            # if iter==2:
                # print(pi[k] * np.power(np.prod(S[k]), -1/2) * np.exp(exp_power))
            r[:,k] = pi[k] * np.power(np.prod(S[k]), -1/2) * np.exp(exp_power)
        r_total = np.sum(r, axis=1)
        r = r / r_total[:,None]
        loss[iter] = -np.sum(np.log(r_total + EPSILON))

        if iter > 1 and abs(loss[iter] - loss[iter-1]) <= TOLERANCE * abs(loss[iter]):
            break
        
        r_total_i_wise = np.sum(r, axis=0)
        pi = r_total_i_wise / N
        mu = np.dot(r.T, X) / r_total_i_wise[:,None]
        S = np.dot(r.T, X ** 2) / r_total_i_wise[:,None] - mu ** 2 + EPSILON

    return pi, mu, S, loss

transform = transforms.Compose([transforms.ToTensor()])


if __name__ == "__main__":
    data_train = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)
    idx = data_train.targets == 0
    np_X = data_train.data[idx].numpy()
    N, d1, d2 = np_X.shape
    X = np_X.reshape(N, d1*d2)
    # pi, mu, S, loss = GMM(X, 5)

    res = [None] * 5
    with Pool() as pool:
        res = pool.starmap(GMM, [[X, k] for k in range(5)])

    print(res[4])


