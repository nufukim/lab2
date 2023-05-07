import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {"depth":100, "purity_bound":5e-2, "gainfunc":"gainratio"}

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: n, 标签向量
    @return: 熵
    """
    # TODO: YOUR CODE HERE
    n = len(Y)
    ufeat, featcnt = np.unique(Y, return_counts=True)
    featp = featcnt / n
    ans = 0
    for x in featp:
        ans = ans - x*np.log(x)
    return ans


def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    feat = X[:, idx]
    ufeat = np.unique(feat).tolist()
    # TODO: YOUR CODE HERE
    n = len(Y)
    m = len(ufeat)
    Y_a = [[] for i in range(m)]
    for i in range(n):
        Y_a[ufeat.index(X[i][idx])].append(Y[i])
    ret = entropy(Y)
    for Y_i in Y_a:
        ret = ret - len(Y_i)/n*entropy(Y_i)
    return ret


def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: n, 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return:
        """
        return len(self.children) == 0


def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr=""):
    root = Node()
    u, ucnt = np.unique(Y, return_counts=True)
    root.label = u[np.argmax(ucnt)]
    # print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") #可用于debug
    # 当达到终止条件时，返回叶节点
    # TODO: YOUR CODE HERE
    if len(u) == 1 or depth == 0 or entropy(Y) < purity_bound:
        return root
    gains = [gainfunc(X, Y, i) for i in unused]
    idx = np.argmax(gains)
    root.featidx = unused[idx]
    unused = deepcopy(unused)
    unused.pop(idx)
    feat = X[:, root.featidx]
    ufeat = np.unique(feat).tolist()
    # 按选择的属性划分样本集，递归构建决策树
    # 提示：可以使用prefixstr来打印决策树的结构
    # TODO: YOUR CODE HERE
    n = len(Y)
    X_son = [[] for i in range(len(ufeat))]
    Y_son = [[] for i in range(len(ufeat))]
    for i in range(n):
        X_son[ufeat.index(feat[i])].append(X[i])
        Y_son[ufeat.index(feat[i])].append(Y[i])
    for i in range(len(ufeat)):
        root.children[ufeat[i]] = buildTree(np.array(X_son[i]), np.array(Y_son[i]), unused, depth-1, purity_bound, gainfunc)
    return root


def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

