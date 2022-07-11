# %%
import numpy as np

MAX_ITER = 3
EPSILON = 0.001
TOLERANCE = 1e-5

# %%
def initializeModel(K, d):
    np.random.seed(0)
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)

    np.random.seed(0)
    mu = np.random.normal(0, 2, size=(K, d))

    np.random.seed(0)
    S = np.random.rand(K, d) + 0.5

    return pi, mu, S



# %%
print(initializeModel(4, 3))
a, b, c = initializeModel(4,3)

# %%
def GMM(X, K_RANGE):
    N, d = X.shape
    pi, mu, S = initializeModel(K_RANGE, d)
    r = np.zeros((N, K_RANGE))
    loss = [0] * MAX_ITER

    for iter in range(MAX_ITER):
        for k in range(K_RANGE):
            exp_power = np.dot((X-mu[k]) ** 2, 1/S[k]) * (-1/2)
            # if iter==2:
                # print(pi[k] * np.power(np.prod(S[k]), -1/2) * np.exp(exp_power))
            r[:,k] = pi[k] * np.power(np.prod(S[k]), -1/2) * np.exp(exp_power)
            # np.exp((X[0]-mu[k]).T.dot(np.linalg.inv(np.diag(S[k]))).dot(X[0]-mu[k]) * -1/2) * pi[k] * np.power(np.prod(S[k]), -0.5)

        r_total = np.sum(r, axis=1)
        r = r / r_total[:,None]
        loss[iter] = -np.sum(np.log(r_total + EPSILON))

        if iter > 1 and abs(loss[iter] - loss[iter-1]) <= TOLERANCE * abs(loss[iter]):
            break
        
        r_total_i_wise = np.sum(r, axis=0)
        pi = r_total_i_wise / N
        mu = np.dot(r.T, X) / r_total_i_wise[:,None]
        S = np.dot(r.T, X ** 2) / r_total_i_wise[:,None] - mu ** 2
        for k in range(K_RANGE):
            ans = 0
            for i in range(N):
                ans += r[i][k] * X[i][:,None].dot(X[i][:,None].T)
            ans = ans / r_total_i_wise[k] - mu[k][:,None].dot(mu[k][:,None].T)

    return pi, mu, S, loss

# %%
X = np.loadtxt(open("./A4/gmm_dataset.csv", "rb"), delimiter=",")

# X = np.array(
#     [
#     [1,2,2,1,2],
#     [4,2,1,2,3],
#     [4,3,1,2,3],
#     [3,1,2,3,4]
#     ]
# )
# for k in range(10):
pi, mu, S, loss = GMM(X, 3)
print(loss[-1])
