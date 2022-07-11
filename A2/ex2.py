import numpy as np
from statsmodels.discrete.discrete_model import Logit
from sklearn.svm import SVC

#Exercise 2
#Usage: python3 ex2.py
#Load the files in some other way (e.g., pandas) if you prefer
X_train_A = np.loadtxt('data/X_train_A.csv', delimiter=",")
Y_train_A = np.loadtxt('data/Y_train_A.csv',  delimiter=",").astype(int)

soft_svc = SVC(C=1, kernel="linear")
soft_svc.fit(X_train_A, Y_train_A)
print(f"soft margin: {soft_svc.coef_}")
hard_svc = SVC(C=float('inf') , kernel="linear")
hard_svc.fit(X_train_A, Y_train_A)
print(f"hard margin: {hard_svc.coef_}")
print(f"difference: {soft_svc.coef_ - hard_svc.coef_}")

# PROBLEMATIC METHOD
# logit = Logit(Y_train_A, X_train_A).fit()

result = np.dot(X_train_A, soft_svc.coef_.T).flatten()
# print(result)
# print(len([i for i in result if i <= 1]))
# print(len([i for i in result if np.isclose(i, 1.0) or np.isclose(i, -1.0)]))

print(f"alpha: {soft_svc.dual_coef_}")
print(f"support vectors: {soft_svc.support_}")
print(f"difference: {np.dot(soft_svc.dual_coef_, soft_svc.support_vectors_) - soft_svc.coef_}")

X_train_B = np.loadtxt('data/X_train_B.csv', delimiter=",")
Y_train_B = np.loadtxt('data/Y_train_B.csv', delimiter=",").astype(int)
X_test_B = np.loadtxt('data/X_test_B.csv', delimiter=",")
Y_test_B = np.loadtxt('data/Y_test_B.csv', delimiter=",").astype(int)

soft_svc = SVC(C=1, kernel="linear")
soft_svc.fit(X_train_B, Y_train_B)
print(f"soft margin: {soft_svc.coef_}")
# print(f"alpha: {soft_svc.dual_coef_}")
print(f"number of support vectors: {len(soft_svc.support_)}")
# print(f"difference: {np.dot(soft_svc.dual_coef_, soft_svc.support_vectors_) - soft_svc.coef_}")

logit = Logit(Y_train_B, X_train_B).fit()
# PROBLEMATIC METHOD
# hard_svc = SVC(C=float('inf') , kernel="linear")
# hard_svc.fit(X_train_B, Y_train_B)

# 0-1 loss
soft_svc_predicted = soft_svc.predict(X_test_B)
n, d = X_test_B.shape
counter, error = 0, 0
for predicted, actual in zip(soft_svc_predicted, Y_test_B):
    if not np.isclose(predicted, actual):
        error += 1
print(f"soft SVC loss: {error, error / n}")

logit_predicted = logit.predict(X_test_B)
error = 0
for predicted, actual in zip(logit_predicted, Y_test_B):
    counter += 1
    if not np.isclose(predicted, actual):
        error += 1
print(f"logistic regression loss: {error, error / n}")