#  Polynomial_Regression
import matplotlib.pyplot as plt
import numpy as np


def f( x ):
    """予測関数

    :param x:予測データ
    :return: 予測された値
    """
    return np.dot(x, theta)


def E( x, y ):
    """目的関数、誤差（最小にしたい）

    :param x: 学習データ
    :param y: 学習データ
    :return: 目的関数の値
    """
    return 0.5 * np.sum((y - f(x)) ** 2)


def standardize( x ):
    """学習データの平均を0，分散を1にする

    :param x: 学習データ
    :return: 標準化されたデータ
    """
    return (x - mu) / sigma


def MSE( x, y ):
    """平均二乗誤差を求める
    """
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)


def update():
    """パラメータを更新する

    :return: None
    """
    EPS = 10 ** -6  # 終了条件：誤差がこれより小さくなったら終了する
    ETA = 10 ** -3  # 学習率
    diff = 1
    global theta
    errors.append(MSE(X, train_y))
    while diff > EPS:
        #  学習データを並べるためにランダムな順列を用意
        p = np.random.permutation(X.shape[0])
        #  学習データをランダムに取り出して、確率的勾配法でパラメータを更新
        for x, y in zip(X[p, :], train_y[p]):
            theta = theta - ETA * (f(x) - y) * x
        #  誤差の計算
        errors.append(MSE(X, train_y))
        diff = errors[-2] - errors[-1]


def to_matrix( x ):
    """学習データの行列を作成

    :param x: 学習データ
    :return: 学習データの行列
    """
    temp = [np.ones(x.shape[0])]
    for i in range(1, N):
        temp.append(x ** i)
    return np.vstack(temp).T


train = np.loadtxt("click.csv", delimiter=",", skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

#  標準化 -> 収束が早くなる
mu = train_x.mean()
sigma = train_x.std()
train_z = standardize(train_x)

#  ----------  多項式回帰  ----------
N = 3  # N次の多項式

theta = np.random.rand(N)
X = to_matrix(train_z)

#  ----------  平均二乗誤差  ----------
errors = []  # 誤差の履歴

update()

x_axis = np.linspace(-3, 3, 100)
#  多項式回帰
plt.plot(train_z, train_y, "o")
plt.plot(x_axis, f(to_matrix(x_axis)))

# MSEを表示
# plt.plot(np.arange(len(errors)), errors)
plt.show()
