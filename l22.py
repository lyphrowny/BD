from sklearn.linear_model import Ridge, Lasso
from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_poly(x, *, degree=11):
    return np.array([x ** (i + 1) for i in range(degree)]).T


def task():
    x, y = map(np.array, zip((-2, -7), (-1, 0), (0, 1), (1, 2), (2, 9)))
    models = {
        "ridge": Ridge,
        "lasso": Lasso
    }
    x_test = np.linspace(-2, 2, 100)

    alphas = (1, .1, .01, .001)

    matplotlib.use('TkAgg')

    fig, axs = plt.subplots(len(models), len(alphas))
    for j, (model_name, model) in enumerate(models.items()):
        print(model_name)
        for i, alpha in enumerate(alphas):
            name = f"{model_name}, alpha({alpha})"
            lm = model(alpha)
            lm.fit(get_poly(x), y)
            axs[j, i].plot(x_test, lm.predict(get_poly(x_test)))
            axs[j, i].plot(x, y, "o")
            axs[j, i].set_title(name)
            axs[j, i].grid()
            print(f"\talpha({alpha}): {' '.join(map(lambda _: f'{_:.6f}', lm.coef_))}")
    print()
    plt.show()

    sigmas = (.1, .2, .3)
    opt_alpha = 0.01
    fig, axs = plt.subplots(len(models), len(sigmas))

    for j, (model_name, model) in enumerate(models.items()):
        print(model_name)
        for i, sigma in enumerate(sigmas):
            name = f"{model_name}, sigma: {sigma}"
            lm = model(opt_alpha)
            lm.fit(get_poly(x), y + stats.norm.rvs(size=len(y), scale=sigma))
            axs[j, i].plot(x_test, lm.predict(get_poly(x_test)))
            axs[j, i].plot(x, y, "o")
            axs[j, i].set_title(name)
            axs[j, i].grid()
            print(f"\tsigma({sigma}): {' '.join(map(lambda _: f'{_:.6f}', lm.coef_))}")
    plt.show()


if __name__ == "__main__":
    task()
