from operator import methodcaller
import scipy.stats as stats
import numpy as np


def huber(distr, k=1.44):
    return np.mean(np.where(np.abs(distr) > k, k * np.sign(distr), distr))
    # return x if abs(x) <= k else k * np.sign(x)


def two_step(distr):
    distr = np.sort(distr)

    def find_outliers():
        q14, q34 = np.quantile(distr, (.25, .75), method="closest_observation")
        wid = q34 - q14
        x_l, x_u = max(distr[0], q14 - 3 / 2 * wid), min(distr[~0], q34 + 3 / 2 * wid)
        # bxplt_outliers = distr[np.logical_or(distr < x_l, distr > x_u)]
        return x_l, x_u

    x_l, x_u = find_outliers()
    return np.mean(distr[np.logical_and(x_l <= distr, distr <= x_u)])


def monte_carlo(distrs, size, n=10_000):
    for dname, distr in distrs.items():
        print(dname)
        for name, meth in zip("mean median huber two_step".split(), (np.mean, np.median, huber, two_step)):
            means = [meth(distr(size)) for _ in range(n)]
            print(f"\t{name}:\n"
                  f"\t\tmean: {np.mean(means):.6f}\n"
                  f"\t\tvariance: {np.std(means):.6f}")
        print()


if __name__ == "__main__":
    _distr = lambda n, d="norm": methodcaller(d, 0, 1)(stats).rvs(size=n)
    _norm = lambda n: _distr(n=n, d="norm")
    _cauchy = lambda n: _distr(n=n, d="cauchy")
    _comb = lambda n: .9 * _norm(n) + .1 * _cauchy(n)

    n = 100
    distrs = {
        "normal": _norm,
        "cauchy": _cauchy,
        "combination": _comb
    }

    monte_carlo(distrs, n)
