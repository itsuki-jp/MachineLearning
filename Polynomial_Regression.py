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


def update():
    """パラメータを更新する

    :return: None
    """
    EPS = 10 ** -2  # 終了条件：誤差がこれより小さくなったら終了する
    ETA = 10 ** -3  # 学習率
    diff = 1
    count = 0  # 何回更新したか
    global theta
    error = E(X, train_y)
    while diff > EPS:
        #  パラメータを更新
        theta = theta - ETA * np.dot(f(X) - train_y, X)
        #  誤差の計算
        current_error = E(X, train_y)
        diff = error - current_error
        error = current_error

        count += 1


def to_matrix( x ):
    """学習データの行列を作成

    :param x: 学習データ
    :return: 学習データの行列
    """
    temp = [np.ones(x.shape[0])]
    for i in range(1, N):
        temp.append(x ** i)
    return np.vstack(temp).T


train = np.loadtxt("files/click.csv", delimiter=",", skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

#  標準化 -> 収束が早くなる
mu = train_x.mean()
sigma = train_x.std()
train_z = standardize(train_x)

#  ----------  多項式回帰  ----------
N = 3  # パラメータ数

theta = np.random.rand(N)
X = to_matrix(train_z)

update()

x_axis = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, "o")
plt.plot(x_axis, f(to_matrix(x_axis)))
plt.show()
