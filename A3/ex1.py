import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

MAX_DEPTH = 20

class DecisionTree:
    #You will likely need to add more arguments to the constructor
    def __init__(self, depth, loss_function, feature_indices_used=None):
        #Implement me!
        self.left = None
        self.right = None
        self.depth = depth
        self.loss_function = loss_function
        self.feature = None
        self.threshold = None
        self.p_hat = -1
        self.feature_indices_used = feature_indices_used
        return

    def findBestFeature(self, X, y):
        # Sort each feature for splitting
        features = X.transpose().copy()
        features.sort(axis=1)
        d, n = features.shape
        min_loss = float('inf')
        min_feature, min_threshold = None, None
        min_left, min_right = None, None

        features_used = range(d) if self.feature_indices_used is None else self.feature_indices_used
        for feature_i in features_used:
            for value_i in range(n-1):
                split = (features[feature_i][value_i] + features[feature_i][value_i+1]) / 2
                # print(features[feature_i][value_i], features[feature_i][value_i+1], split)
                left_indices = set([idx for idx, feature in enumerate(X) if feature[feature_i] <= split])
                right_indices = set([idx for idx, feature in enumerate(X) if feature[feature_i] > split])
                if not left_indices or not right_indices:
                    continue

                left_X = [x for i, x in enumerate(X) if i in left_indices]
                left_Y = [y for i, y in enumerate(y) if i in left_indices]
                left_loss = self.loss_function(left_X, left_Y)

                right_X = [x for i, x in enumerate(X) if i in right_indices]
                right_Y = [y for i, y in enumerate(y) if i in right_indices]
                right_loss = self.loss_function(right_X, right_Y)

                loss = len(left_X) / len(X) * left_loss + len(right_X) / len(X) * right_loss

                if loss < min_loss:
                    min_loss = loss
                    min_feature = feature_i
                    min_threshold = split
                    min_left = left_indices
                    min_right = right_indices
        return min_feature, min_threshold, min_left, min_right


    def build(self, X, y):
        #Implement me!
        self.p_hat = y.count(1) / len(y)
        if self.depth >= MAX_DEPTH:
            return
        if len(y) <= 0:
            return
        if y.count(y[0]) == len(y):
            return
        
        self.feature, self.threshold, left_indices, right_indices = self.findBestFeature(X, y)
        left_X = np.array([x for i, x in enumerate(X) if i in left_indices])
        left_Y = [y for i, y in enumerate(y) if i in left_indices]

        right_X = np.array([x for i, x in enumerate(X) if i in right_indices])
        right_Y = [y for i, y in enumerate(y) if i in right_indices]

        self.left = DecisionTree(self.depth + 1, self.loss_function)
        self.left.build(left_X, left_Y)
        self.right = DecisionTree(self.depth + 1, self.loss_function)
        self.right.build(right_X, right_Y)
        return
    
    def predict(self, X):
        #Implement me!
        cur_depth_pred = 1 if self.p_hat >= 0.5 else 0
        if not self.left and not self.right:
            return [cur_depth_pred]
        if X[self.feature] <= self.threshold:
            child_pred = self.left.predict(X)
        else:
            child_pred = self.right.predict(X)
        
        return [cur_depth_pred] + child_pred

        # For print purpose
    def __str__(self, level=0, contain="root", prev_word=-1, prev_thre = -1):
        ret = "|" + "\t"*level + contain + ": "  + str(prev_word) + " " + str(prev_thre) + ". current feature: " + str(self.feature) + " " + str(self.threshold) + " depth: " + str(self.depth) + "\n"

        
        if self.left and self.right:
            ret += self.left.__str__(level+1, "less than", self.feature, self.threshold)
            ret += self.right.__str__(level+1, "greater than", self.feature, self.threshold)
        else:
            ret += "|" + "\t"*(level + 1) + "PE: " + str(self.p_hat) + "\n"
        
        return ret

def miss_classification_err(X, y):
    p_hat = y.count(1) / len(y)
    return min(p_hat, 1-p_hat)

def gini_coefficient(X, y):
    p_hat = y.count(1) / len(y)
    return p_hat * (1-p_hat)

def entropy(X, y):
    p_hat = y.count(1) / len(y)
    first = - p_hat * np.log2(p_hat) if p_hat > 0 else 0
    second = - (1 - p_hat) * np.log2(1 - p_hat) if 1 - p_hat > 0 else 0

    return first + second

def getAccuracy(tree, X, Y):
    pred_results = [[] for i in range(len(Y))]
    for i, x in enumerate(X):
        pred_results[i] = tree.predict(x)

    accuracy = np.zeros(MAX_DEPTH)
    for depth in range(MAX_DEPTH):
        for point_idx, y_pred in enumerate(pred_results):
            pred_index = min(depth, len(y_pred) - 1)
            if y_pred[pred_index] == Y[point_idx]:
                accuracy[depth] += 1

    accuracy /= len(Y)
    return accuracy

def question1():
    methods = [miss_classification_err, gini_coefficient, entropy]
    for loss_func in methods:
        tree = DecisionTree(0, loss_func)
        tree.build(X_train, list(y_train))
        train_acc = getAccuracy(tree, X_train, y_train)
        test_acc = getAccuracy(tree, X_test, y_test)

        plt.xlabel("Tree Depth")
        plt.ylabel("Accuracy")
        plt.plot(range(MAX_DEPTH), train_acc, marker='o', label="Train Accuracy")
        plt.plot(range(MAX_DEPTH), test_acc, marker='o', label="Test Accuracy")
        plt.legend(loc="best")
        plt.title(f"Accuracy of Decision Tree with Loss Function {loss_func.__name__}")
        plt.savefig(f"Decision_tree_{loss_func.__name__}")
        # plt.show()
        plt.close()

def getBaggingAccuracy(trees, X, Y):
    pred_results = [-1 for _ in range(len(Y))]
    for data_i, x in enumerate(X):
        local_votes = 0
        for tree_i, tree in enumerate(trees):
            if tree.predict(x)[-1] == 1:
                local_votes += 1
        pred_results[data_i] = 1 if local_votes > len(trees) / 2 else 0

    accuracy = 0
    for y_pred, y_true in zip(pred_results, Y):
        if y_pred == y_true:
            accuracy += 1

    accuracy /= len(Y)
    return accuracy

def TrainBagging(X, y, features_used=None):
    N, _ = X.shape
    random_indices = np.random.choice(N, N)
    bootstrap_x = X[random_indices, :]
    bootstrap_y = y[random_indices]
    tree = DecisionTree(0, entropy, features_used)
    tree.build(bootstrap_x, list(bootstrap_y))
    return tree


if __name__ == '__main__': 
    #Load data
    X_train = np.loadtxt('data/X_train.csv', delimiter=",")
    y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
    X_test = np.loadtxt('data/X_test.csv', delimiter=",")
    y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)
    question1()

    # Bagging
    N, d = X_train.shape
    MAX_DEPTH = 3

    accuracies = [0] * 11
    for acc_i in tqdm(range(len(accuracies))):
        Bagging_Trees = [None] * 101

        with Pool() as pool:
            Bagging_Trees = pool.starmap(TrainBagging, [[X_train, y_train] for _ in range(101)])
        # for i in range(101):
        #     random_indices = np.random.choice(N, N)
        #     # random_indices.sort()
        #     bootstrap_x = X_train[random_indices, :]
        #     bootstrap_y = y_train[random_indices]
        #     # print(bootstrap_y)
        #     # print(random_indices)
        #     tree = DecisionTree(0, entropy)
        #     tree.build(bootstrap_x, list(bootstrap_y))
        #     Bagging_Trees[i] = tree
        #     print(f"finished {i}")

        acc = getBaggingAccuracy(Bagging_Trees, X_test, y_test)
        accuracies[acc_i] = acc
        print(f"Iteration {acc_i} with acc: {acc}")
    print(accuracies)
    print(f"Bagging accuracy data: median: {np.median(accuracies)}, min: {min(accuracies)}, max: {max(accuracies)}")

    # Random Forest
    accuracies = [0] * 11
    for acc_i in range(len(accuracies)):
        Bagging_Trees = [None] * 101

        with Pool() as pool:
            Bagging_Trees = pool.starmap(TrainBagging, [[X_train, y_train, np.random.choice(d, 4, replace=False)] for _ in range(101)])

        acc = getBaggingAccuracy(Bagging_Trees, X_test, y_test)
        accuracies[acc_i] = acc
        print(f"Iteration {acc_i} with acc: {acc}")
    print(accuracies)
    print(f"Random Forest accuracy data: median: {np.median(accuracies)}, min: {min(accuracies)}, max: {max(accuracies)}")