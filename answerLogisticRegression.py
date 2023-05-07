import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 3.7e-1  # 学习率
wd = 5.5e-2  # l2正则化项系数

def sigmoid(x):
    return 1 / (np.exp(-x/100) + 1)

def f(X, weight, bias):
    return np.dot(X, weight) + bias

def predict(X, weight, bias):
    """
    使用输入的weight和bias预测样本X是否为数字0
    @param X: n*d 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: d
    @param bias: 1
    @return: n wx+b
    """
    # TODO: YOUR CODE HERE
    n = len(X)
    haty = np.zeros((n,), dtype=float)
    for i in range(n):
        haty[i] = f(X[i], weight, bias)
    return haty

def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: n*d 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: d
    @param bias: 1
    @param Y: n 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: n 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: 1 由交叉熵损失函数计算得到
        weight: d 更新后的weight参数
        bias: 1 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    n = len(Y)
    d = len(weight)
    haty = np.zeros((n,), dtype=float)
    J_w = np.zeros((d,), dtype=float)
    J_b = 0
    Loss = 0
    for i in range(n):
        haty[i] = f(X[i], weight, bias)
        tmp = sigmoid(Y[i] * haty[i])
        Loss = Loss - np.log(tmp)/n
        for j in range(d):
            J_w[j] = J_w[j] - (1 - tmp) * X[i][j] * Y[i] / 100 / n
        J_b = J_b - (1 - tmp) * Y[i] / n
        
    for j in range(d):
        J_w[j] = J_w[j] + 2 *wd *weight[j]
        Loss = Loss + wd * weight[j] * weight[j]
        weight[j] = weight[j] - lr * J_w[j]
    bias = bias - lr * J_b
    return haty, Loss, weight, bias
