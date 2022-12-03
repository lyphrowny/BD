import numpy as np
import scipy.stats as stats
from sklearn import linear_model
from sklearn import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


def f(x):
    return 1 + 3 * x[0] - 2 * x[1] + x[2] + stats.norm.rvs(size=1)[0]


def rss(y, pred):
    return np.sum((y - pred) ** 2)


def rse(y, pred):
    return np.sqrt(rss(y, pred) / (len(pred) - 2))


def r_sq(y, pred):
    return stats.pearsonr(y, pred).statistic ** 2


def task1():
    n, nn = 3, 20
    vects = np.zeros((nn, n))
    for i in range(nn):
        vects[i] = stats.norm.rvs(size=n)
    # print(vects)
    y_k = np.fromiter(map(f, vects), dtype=np.float32)
    # print(y_k)
    lm = linear_model.LinearRegression()
    lm.fit(X=vects, y=y_k)
    print(f"Model's coeffs: {lm.coef_}")

    pred = lm.predict(X=vects)
    for name, meth in zip("rss rse r**2".split(), (rss, rse, r_sq)):
        print(f"{name}: {meth(y_k, pred):.6f}")
    print()


def scrapper():
    import requests
    from bs4 import BeautifulSoup
    from operator import attrgetter
    site = "http://www.pogodaiklimat.ru/history/27612.htm"

    soup = BeautifulSoup(requests.get(site).content, "html.parser")
    years, temps = soup.find_all("table")[:2]
    years = map(int, map(attrgetter("text"), years.find_all("td")[1:]))
    temps = map(lambda tr: tr.find_all("td")[~0].text, temps.find_all("tr")[1:])
    d = filter(lambda t: t[1] != "999.9", zip(years, temps))

    with open("med_year_temps.txt", "w") as f:
        for y, t in d:
            f.write(f"{y} {t}\n")


def task2():
    # scrapper()
    x, y = map(np.array, zip(*map(lambda l: (lambda y, t: (int(y), float(t)))(*l.split()), open("med_year_temps.txt"))))

    poly_degree = 6
    poly = preprocessing.PolynomialFeatures(degree=poly_degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))

    clf = linear_model.LinearRegression()
    # Build step forward feature selection
    sfs1 = sfs(clf, k_features=x_poly.shape[1], forward=True, floating=False, scoring='r2', cv=5)
    # Perform SFFS
    sfs1 = sfs1.fit(x_poly, y)

    lr_model = linear_model.LinearRegression()
    mX = x_poly[:, sfs1.k_feature_idx_]
    lr_model.fit(X=mX, y=y)
    pred = lr_model.predict(X=mX)

    # pred = lm.predict(X=x_poly)
    print("Poly regression")
    for name, meth in zip("rss rse r**2".split(), (rss, rse, r_sq)):
        print(f"{name}: {meth(y, pred):.6f}")
    #
    # lm = linear_model.LinearRegression()
    # lm.fit(X=x.reshape(-1, 1), y=y)
    # pred = lm.intercept_ + lm.coef_[0] * x
    #
    # for name, meth in zip("rss rse r**2".split(), (rss, rse, r_sq)):
    #     print(f"{name}: {meth(y, pred):.6f}")


if __name__ == "__main__":
    task1()
    task2()
