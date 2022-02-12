#  iris分類のやつ　ロジスティック、確率的勾配降下法

import numpy as np

# 学習データを読み込む
train_data = np.loadtxt('train_data.txt', delimiter=',')
train_x = train_data[:, :4]
train_y = train_data[:, 4]

# パラメータの初期化
theta = np.random.rand(5)


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


X = to_matrix(train_x)

# 学習率
ETA = 1e-3

# 繰り返し回数
epoch = 5000

for _ in range(epoch):
    # 確率的勾配降下法でパラメータ更新
    p = np.random.permutation(X.shape[0])
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x

TF = [0, 0]  # True, False
test_data = np.loadtxt('test_data.txt', delimiter=',')
for t_data in test_data:
    test = t_data[:4]
    res = f(to_matrix(np.array([test])))
    if (res[0] >= 0.5) == t_data[-1]:
        TF[0] += 1
    else:
        TF[1] += 1
print(TF)
