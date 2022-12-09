from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cluster, neighbors, naive_bayes


def _make_cluster(xs, classification):
    clust = defaultdict(list)
    for x, clust_num in zip(xs, classification):
        clust[clust_num].append(x)
    return clust


def _plot(ttl, what):
    for item in what.values():
        plt.scatter(*zip(*item))
    plt.grid()
    plt.title(ttl)
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
    _plot("true", {k: v for k, v in enumerate(_xs)})
    xs = np.array(_xs).reshape(-1, 2)

    cl_m = cluster.KMeans(n_clusters=n_clusters)
    cl_m.fit(xs)
    clust_pred = cl_m.predict(xs)
    _plot("KMeans", _make_cluster(xs, clust_pred))

    y_classes = np.arange(n_clusters)[np.newaxis, ...].repeat(size, axis=1).flatten()
    kn_model = neighbors.KNeighborsClassifier(n_neighbors=5)
    kn_model.fit(X=xs, y=y_classes)
    y_kn = kn_model.predict(xs)
    _plot("KNeighbors", _make_cluster(xs, y_kn))

    nb_model = naive_bayes.GaussianNB()
    nb_model.fit(X=xs, y=y_classes)
    y_nb = nb_model.predict(xs)
    _plot("Bayesian", _make_cluster(xs, y_nb))


if __name__ == "__main__":
    task()
