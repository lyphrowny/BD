from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cluster, neighbors, naive_bayes
from functools import reduce, partial


def _errors(res, orig):
    c = defaultdict(list)
    # otherwise the orig is exhausted after 1 iteration of the outer loop
    orig = list(orig)
    for cl, el in res:
        for i, distr in orig:
            if el in distr and i != cl:
                c[(i, cl)].append(el)
    return c


def __find_orig_order(c, _xs, thres):
    swapped = {}
    for k, v in c.items():
        a, b = [swapped.get(_, _) for _ in k]
        if len(v) > thres and swapped.get(a, None) != b:
            swapped[a] = b
            swapped[b] = a
            _xs[a], _xs[b] = _xs[b], _xs[a]
    return bool(swapped)


def _make_dyn_dict(cl, dist):
    return reduce(lambda g, el: g[el[0]].append(el[1]) or g, zip(cl, dist), defaultdict(list))


def _plot_diff(res_cl, ttl, ttls, *, t_classes, t_dist):
    c = _errors(zip(res_cl, t_dist), zip(t_classes, t_dist))
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
    fig.suptitle(ttl)
    for ax, what, ttl in zip(axs, (_make_dyn_dict(t_classes, t_dist), _make_dyn_dict(res_cl, t_dist), c), ttls):
        for lab, _what in what.items():
            ax.scatter(*zip(*_what), label=lab)
        ax.set_title(ttl)
        ax.legend()
    plt.show()


def task():
    means = (
        (3, 3),
        (9, 2),
        (9, 6)
    )
    *covs, = map(np.diag, ((1.5, 1.5), (1, 1), (1, 1)))
    size = 200
    n_clusters = len(covs)

    matplotlib.use('TkAgg')
    _xs = [stats.multivariate_normal.rvs(size=size, mean=mean, cov=cov) for mean, cov in zip(means, covs)]
    xs = np.array(_xs).reshape(-1, 2)
    c_xs = xs.copy()

    np.random.shuffle(xs)
    _split = int(xs.shape[0] * .7)
    train = xs[:_split]
    test = xs[_split:]
    # classification of test distribution
    cl_test = np.where(test[:, None] == c_xs)[1][::2] // size

    cl_m = cluster.KMeans(n_clusters=n_clusters)
    cl_m.fit(train)
    # prediction to use as target values
    train_classes = cl_m.predict(train)
    test_classes = cl_m.predict(test)

    # remap the _xs classes; find the errors
    while __find_orig_order((c := _errors(zip(cl_test, test), enumerate(_xs))), _xs, len(test) / n_clusters / 2):
        ...
    # print(c.keys(), *map(len, c.values()))

    plot_diff = partial(_plot_diff, t_classes=test_classes, t_dist=test)
    plot_diff(test_classes, "Test", "orig KMeans err".split())

    for model, ttl, ttls in zip(
            (neighbors.KNeighborsClassifier(n_neighbors=5), naive_bayes.GaussianNB()),
            "KNeighbors Bayes".split(),
            map(lambda s: s.split(), ("KMeans KNeighbors err", "KMeans Bayes err"))
    ):
        model.fit(X=train, y=train_classes)
        plot_diff(model.predict(test), ttl, ttls)


if __name__ == "__main__":
    task()
