import torch
from torch import nn
from d2l import torch as d2l
import random


def synthetic_data(w, b, num_examples):
    # 生成 y = Xw + b + 噪声
    X = torch.normal(0, 1, (num_examples, len(w)))
    """
    torch.normal(mean, std, size)
    该函数返回从单独的正态分布中提取的随机数的张量，该正态分布的均值是mean，标准差是std
    """
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    data_size = features.shape[0]
    index = list(range(data_size))
    random.shuffle(index)
    for i in range(0, data_size, batch_size):
        batch_index = torch.tensor(
            index[i:min(i + batch_size, data_size)]
        )
        yield features[batch_index], labels[batch_index]


def liner(w, b, features):
    return torch.matmul(features, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2


def sgd(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


data_size = 1000
batch_size = 10
w_true = torch.tensor([3, -1.2, 0.23])
b_true = 5
epochs_num = 5
features, labels = synthetic_data(w_true, b_true, data_size)

w = torch.normal(0, 0.01, (3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.03
net = liner
loss = squared_loss

for epoch in range(epochs_num):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(w, b, X), y)
        l.mean().backward()
        sgd([w, b], lr)
    with torch.no_grad():
        train_loss = torch.mean(loss(net(w, b, features), labels))
        print(f'epoch {epoch + 1}, loss is {"%.8f"%train_loss}')