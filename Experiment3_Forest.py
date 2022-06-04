import numpy as np
import math
import random
import matplotlib.pyplot as plt
from typing import Union
import sklearn.ensemble as se
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 结点类型
class DecisionNode(object):
    def __init__(self, f_idx=None, threshold=None, value=None):
        self.f_idx = f_idx
        self.threshold = threshold
        self.value = value
        self.L = None
        self.R = None
        pass
    pass


class MetaLearner(object):
    def __init__(self,
                 min_samples: int = 1,
                 min_gain: float = 0,
                 max_depth: Union[int, None] = None,
                 max_leaves: Union[int, None] = None):
        self.head = None
        self.min_samples = min_samples
        self.min_gain = min_gain
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # 将X 和 y合并，即X的维度为(n, m+1)，第m+1列即为类别
        y = [[i] for i in y]
        y = np.array(y)
        X = np.hstack((X, y))
        self.head = TreeGenerate(X, self.min_samples, self.min_gain)
        return self
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        result = []
        for element in X:
            result.append(self.predictOne(element))
        return np.array(result, dtype=np.int64)
        pass

    def predictOne(self, signle_X):
        node = self.head
        while not (node.L == None and node.R == None):
            if signle_X[node.f_idx] < node.threshold:
                node = node.L
            else:
                node = node.R
        return node.value
    pass


# 返回y中的均值
def ReturnMean(y) -> float:
    y_list = list(y)
    if len(y_list) == 0:
        return 0.0
    y_sum = 0.0
    for i in y_list:
        y_sum += i
    return y_sum / len(y_list)


# 给定数据集的行数，返回其数据集方差
def ReturnInformationVar(X: np.ndarray) -> float:
    if len(X) == 0:
        return 0.0
    y_list = list(X[:, -1])
    y_mean = ReturnMean(X[:, -1])
    y_var = 0.0
    for i in y_list:
        y_var += (i - y_mean) * (i - y_mean)
    return y_var / len(X)



# 从一共m个属性随机抽取k个属性，再从这k个属性中划分出最优属性, 并将最优属性在X的下标返回回去
def FindBestAttribute(X: np.ndarray, min_gain):
    best_gain = 0.0            # 最大信息增益
    best_idx = 0              # 最大信息增益属性的对应的下标，初始化为0
    best_threshold = 0.0
    k = int(math.log(X.shape[1] - 1, 2) + 1)    # 从X.shape[1]个属性中随机抽取k个属性
    wait_extract_list = random.sample([i for i in range(0, X.shape[1] - 1)], k)
    # 接下来对每一个属性求信息增益
    for i in wait_extract_list:
        temp_threshold, temp_gain = ReturnBestThresholdAndVarDivergence(X, i)
        if temp_gain > best_gain and math.fabs(temp_gain - best_gain) > min_gain:
            best_gain = temp_gain
            best_idx = i
            best_threshold = temp_threshold
    return best_idx, best_threshold


# 根据数据集和属性下标，返回该属性的最优划分点和方差增益
def ReturnBestThresholdAndVarDivergence(X: np.ndarray, f_idx: int):
    y_var = ReturnInformationVar(X)
    thresholds = list(set(X[:, f_idx]))
    best_gain = -math.inf
    best_threshold = None
    for threshold in thresholds:
        less_threshold, more_threshold = split(X, f_idx, threshold)
        less_weight = len(less_threshold) / len(X)
        more_weight = len(more_threshold) / len(X)
        y_var_sum = less_weight * ReturnInformationVar(less_threshold) + more_weight * ReturnInformationVar(more_threshold)
        gain = y_var - y_var_sum
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    return best_threshold, best_gain

# 根据数据集和属性下标和划分点来划分数据集
def split(X: np.ndarray, f_idx: int, threshold):
    less_threshold = []
    more_threshold = []
    for (idx, d) in enumerate(X[:, f_idx]):
        if d < threshold:
            less_threshold.append(idx)
        else:
            more_threshold.append(idx)
    return X[less_threshold], X[more_threshold]


def TreeGenerate(X: np.ndarray, min_samples, min_gain) -> DecisionNode:
    global Count
    Count = 0
    if len(set(X[:, -1])) == 1:
        node = DecisionNode(value=list(set(X[:, -1]))[0])
        return node
    elif X.shape[1] == 1:
        node = DecisionNode(value=ReturnMean(X[:, -1]))
        return node
    else:
        best_idx, best_threshold = FindBestAttribute(X, min_gain)
        node = DecisionNode(f_idx=best_idx, threshold=best_threshold)
        less_data = X[np.where(X[:, best_idx] < best_threshold)]
        more_data = X[np.where(X[:, best_idx] >= best_threshold)]
        if len(less_data) == 0:
            node.L = DecisionNode(value=ReturnMean(X[:, -1]))
        else:
            Count += 1
            if Count > min_samples:
                node.L = DecisionNode(value=ReturnMean(X[:, -1]))
            else:
                node.L = TreeGenerate(less_data, min_samples, min_gain)
            Count -= 1
        if len(more_data) == 0:
            node.R = DecisionNode(value=ReturnMean(X[:, -1]))
        else:
            Count += 1
            if Count > min_samples:
                node.R = DecisionNode(value=ReturnMean(X[:, -1]))
            else:
                node.R = TreeGenerate(more_data, min_samples, min_gain)
            Count -= 1
    return node
    pass

def ReturnMeanNumberInOneCol(X: np.ndarray):
    X = X.reshape((len(X), 1))
    List = list(X[:, -1])
    return np.mean(List)


if __name__ == "__main__":
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    print("The shape of X_train is:", X_train.shape)
    print("The shape of y_train is:", y_train.shape)
    print("The shape of X_test is:", X_test.shape)
    print("The shape of y_test is:", y_test.shape)

    my_Forest = []   # 创建森林列表,其中存放的元素是MetaLearner类型
    sklearn_Forest = []
    Number_Tree = np.arange(1, 101)   # 森林中决策树的数量,从1到100棵
    my_var_list = []
    sklearn_var_list = []
    arr_list = [i for i in range(len(X_train))]
    for value in range(len(Number_Tree)):                 # 从第一棵树到100棵树遍历
        for i in range(Number_Tree[value]):
            Sample_list = []                              # 存放抽取到的属性
            Class_list = []                               # 存放对应 抽取到的属性 的 类别
            for j in range(len(arr_list)):
                samples = random.sample(arr_list, 1)      # 不放回的抽取属性的下标
                Sample_list.append(X_train[samples][0])
                Class_list.append(y_train[samples][0])
            Sample_array = np.array(Sample_list)
            Class_array = np.array(Class_list)
            ML = MetaLearner()
            rfr = se.RandomForestRegressor()
            my_Forest.append(ML.fit(X=Sample_array, y=Class_array))
            sklearn_Forest.append(rfr.fit(X=Sample_array, y=Class_array))

        my_result = []
        sklearn_result = []
        for i in range(Number_Tree[value]):
            my_result.append(my_Forest[i].predict(X_test))
            sklearn_result.append(sklearn_Forest[i].predict(X_test))
        my_result_array = np.array(my_result)
        sklearn_result_array = np.array(sklearn_result)

        my_finally_result = []
        sklearn_finally_result = []
        for i in range(my_result_array.shape[1]):
            my_finally_result.append(ReturnMeanNumberInOneCol(my_result_array[:, i]))

        for i in range(sklearn_result_array.shape[1]):
            sklearn_finally_result.append(ReturnMeanNumberInOneCol(sklearn_result_array[:, i]))

        # print("The len of finally_result is:", len(my_finally_result))
        # print("The len of sklearn_result is:", len(sklearn_finally_result))
        # print("The len of y_test is:", len(y_test))
        # print("finally_result is:", list(my_finally_result))
        # print("sklearn_result is:", list(sklearn_finally_result))
        # print("y_test is:", [int(i) for i in list(y_test)])

        # 算出均方误差
        my_var_rate = 0.0
        sklearn_var_rate = 0.0
        for i in range(len(y_test)):
            my_var_rate += (my_finally_result[i] - y_test[i]) ** 2
            sklearn_var_rate += (sklearn_finally_result[i] - y_test[i]) ** 2
        print("the my_var_rate is:", my_var_rate / len(y_test))
        print("the sklearn_var_rate is:", sklearn_var_rate / len(y_test))
        my_var_list.append(my_var_rate / len(y_test))
        sklearn_var_list.append(sklearn_var_rate / len(y_test))

    # 绘图
    plt.plot(Number_Tree, my_var_list, label="my_forest")
    plt.title("my_forest")
    plt.show()

    plt.plot(Number_Tree, sklearn_var_list, label="sklearn_forest")
    plt.title("sklearn_forest")
    plt.show()