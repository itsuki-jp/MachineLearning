```python
import numpy as np
import matplotlib.pyplot as plt


def f( x ):
    """予測関数（1次関数）

    :param x:予測データ
    :return: 予測された値
    """
    return theta0 + theta1 * x


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
    global theta0, theta1
    EPS = 10 ** -2  # 終了条件：誤差がこれより小さくなったら終了する
    ETA = 10 ** -3  # 学習率
    diff = 1
    count = 0  # 何回更新したか

    error = E(train_z, train_y)
    while diff > EPS:
        #  パラメータの更新式
        temp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
        temp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
        #  更新する
        theta0 = temp0
        theta1 = temp1

        current_error = E(train_z, train_y)
        diff = error - current_error
        error = current_error

        count += 1
        print(f"{count} 回目: theta0 = {theta0}, theta1 = {theta1}, diff = {diff}")


train = np.loadtxt("files/click.csv", delimiter=",", skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

#  予測関数で用いいるパラメータを定義
theta0 = np.random.rand()
theta1 = np.random.rand()

#  標準化 -> 収束が早くなる
mu = train_x.mean()
sigma = train_x.std()
train_z = standardize(train_x)

update()


X = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, "o")
plt.plot(X, f(X))
plt.show()

```
