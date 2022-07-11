from re import L
import numpy as np
from matplotlib import pyplot as plt 

X = np.loadtxt(open("data/spambase_X.csv", "rb"), delimiter=",")
y = np.loadtxt(open("data/spambase_y.csv", "rb"), delimiter=",")
n = len(X)
d = len(X[0])
w = np.zeros(d)
b = 0
max_pass = 500
mistake = [0] * max_pass

for t in range(max_pass):
    mistake[t] = 0
    for i in range(n):
        if y[i]*(np.dot(X[i], w) + b) <= 0:
            w = w + y[i] * X[i]
            b = b + y[i]
            mistake[t] += 1

plt.title("Mistakes v.s. Passes")
plt.xlabel("Pass Number")
plt.ylabel("Mistakes")
plt.plot(range(max_pass), mistake)
plt.savefig("Exercise1.png")
# plt.show()
plt.close()