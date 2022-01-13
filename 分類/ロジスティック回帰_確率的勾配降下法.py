#  iris分類のやつ　ロジスティック、確率的勾配降下法

import numpy as np

# 学習データを読み込む
data = np.loadtxt('data.txt', delimiter=',')
train_x = data[:, :4]
train_y = data[:, 4]

# パラメータの初期化
theta = np.random.rand(5)


# x0を加える
def to_matrix( x ):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])


X = to_matrix(train_x)


# シグモイド関数
def f( x ):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# 分類関数
def classify( x ):
    return (f(x) >= 0.5).astype(np.int)


# 学習率
ETA = 1e-3

# 繰り返し回数
epoch = 5000

# 学習を繰り返す
for _ in range(epoch):
    # 確率的勾配降下法でパラメータ更新
    p = np.random.permutation(X.shape[0])
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x

test = np.array([5.0, 3.6, 1.4, 0.2])
res = f(to_matrix(np.array([test])))
print(res)
print("versicolor" if res >= 0.5 else "setosa")
