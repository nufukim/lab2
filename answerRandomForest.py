from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 20     # 树的数量
ratio_data = 0.5   # 采样的数据比例
ratio_feat = 0.5 # 采样的特征比例
hyperparams = {"depth":5, "purity_bound":1e-2, "gainfunc":gain} # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    n = len(X)
    m = len(X[0])
    roots = []
    for i in range(num_tree):
        rands = np.random.uniform(low=0, high=1, size=n)
        X_ = np.array([X[i] for i in range(n) if rands[i] <= ratio_data])
        Y_ = np.array([Y[i] for i in range(n) if rands[i] <= ratio_data])
        rands = np.random.uniform(low=0, high=1, size=m)
        unused = [i for i in range(m) if rands[i] <= ratio_feat]
        roots.append(buildTree(X_, Y_, unused, 25, 1e-1, negginiDA))
    return roots  

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
