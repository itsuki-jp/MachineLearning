import numpy as np
import matplotlib.pyplot as plt


# x0を加える
def to_matrix( x ):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])


# シグモイド関数
def f( x ):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# 分類関数
def classify( x ):
    return (f(x) >= 0.5).astype(np.int)


# データの読み込み
data = np.loadtxt('data.txt', delimiter=',')
train_x = data[:, :4]
train_y = data[:, 4]

# パラメータの初期化
theta = np.random.rand(5)

X = to_matrix(train_x)

ETA = 1e-3  # 学習率

epoch = 5000  # 繰り返し回数

for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

test = np.array([5.0, 3.6, 1.4, 0.2])
res = f(to_matrix(np.array([test])))
print(res)
print("versicolor" if res >= 0.5 else "setosa")
