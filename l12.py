import numpy as np
import matplotlib.pyplot as plt


def task():
    distr = np.random.normal(size=195)
    distr = np.sort(np.hstack((distr, [5, -4, 3.3, 2.99, -3])))
    med, std = np.median(distr), np.std(distr)
    print(f"median: {med:.4f}, std: {std:.4f}")
    print(f"distr edges: {distr[:3]} {distr[~2:]}")
    outliers = distr[np.abs(distr - med) > 3 * std]
    print(f"Outliers: {outliers}")

    # print()
    # grubbs = np.abs(distr[0]-med)/std
    # print(f"Grubbs(1): {grubbs:.4f}")
    # for a in (.01, .05, .1):
    #     print(f"a: {a:.2f}; Grubbs >= a: {grubbs >= a}")

    print()
    q14, q34 = np.quantile(distr, (.25, .75), method="closest_observation")
    wid = q34 - q14
    x_l, x_u = max(distr[0], q14 - 3 / 2 * wid), min(distr[~0], q34 + 3 / 2 * wid)
    print(f"LQ: {q14:.4f}, UQ: {q34:.4f}, IQR: {wid:.4f}")
    print(f"X_L: {x_l:.4f}, X_U: {x_u:.4f}")
    # bxplt_outliers = distr[np.logical_or(distr < x_l, distr > x_u)]
    print(f"Outliers: {distr[distr < x_l]} {distr[distr > x_u]}")

    plt.boxplot(distr, vert=False)
    plt.show()


if __name__ == "__main__":
    task()
