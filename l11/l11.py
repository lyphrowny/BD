import numpy as np
import matplotlib.pyplot as plt
import scipy


def prony(m, xs):
    n = len(xs)
    # generate windows of size `m` and flip them horizontally
    # dropping the last one, as the last equation starts from `n-1`, however
    # the window's result ends with `n` -- bad
    mat = np.fliplr(np.lib.stride_tricks.sliding_window_view(xs[:~0], m))
    alphas, *_ = np.linalg.lstsq(mat, -xs[m:], rcond=None)
    # add 1 as the coef to the `z^m`
    zs = np.roots(np.hstack(([1], alphas)))
    # print("roots\n", zs, "\n")

    mat2 = np.array([zs ** (k - 1) for k in range(1, n + 1)])
    # print(mat2.shape)
    hs, *_ = np.linalg.lstsq(mat2, xs, rcond=None)
    # print("hs\n", hs, "\n")

    # print(np.abs(zs))
    lambdas = np.log(np.abs(zs))
    # print(lambdas)
    omegas = np.arctan(zs.imag / zs.real) / (2 * np.pi)
    As = np.abs(hs)
    phis = np.arctan(hs.imag / hs.real)
    return lambdas, omegas, As, phis


def scrapper():
    from itertools import product, starmap, chain, islice
    import requests
    from bs4 import BeautifulSoup
    from operator import attrgetter
    site = "http://www.pogodaiklimat.ru/monitor.php?id=26063&month={}&year={}"

    d = []
    for (m, y) in chain.from_iterable(
            starmap(product, ((range(9, 13), (2020,)), (range(1, 13), (2021,)), (range(1, 9), (2022,))))):
        soup = BeautifulSoup(requests.get(site.format(m, y)).content, "html.parser")
        *meds, = filter(None, map(lambda tr: tr.find_all("td")[2].text, soup.find("table").find_all("tr")[2:]))
        d.append(meds)

    with open("med_temps.txt", "w") as f:
        for m in d:
            f.write(" ".join(m))
            f.write("\n")


def read_data(fname="l11/med_temps.txt"):
    from operator import methodcaller as m
    return np.array(sum(map(m("split"), open(fname)), []), dtype=float)


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    return (b1 := SS_xy / SS_xx), m_y - b1 * m_x


def hurst(xs):
    rang = np.arange(1, len(xs) + 1)
    ran = np.arange(len(xs))
    ra = np.arange(2, len(xs))

    E = np.vectorize(lambda n: np.mean(xs[:n + 1]))(ran)  # 1..n
    s = np.vectorize(lambda n: np.std(xs[:n + 1]))(ran)  # 1..n
    cs = np.cumsum(xs)  # 1..n

    _X = np.tile(cs, (len(xs), 1)).T - (rang * (np.tile(E, (len(rang), 1))).T).T
    # X = lambda k, n: cs[k]-(k+1)*E[n]

    R = np.vectorize(lambda n: (lambda r: max(r) - min(r))(_X[:n, n]))
    ratio = np.vectorize(lambda n: R(n) / s[n])
    y = lambda n: np.log(ratio(n))
    z = np.log

    Z = z(ra)
    Y = y(ra)
    H, c = estimate_coef(Z, Y)
    print(f"H: {H:.4f} -> {'random persistent antipersistent'.split()[.53 < H < 1 or 2 * (0 < H < .47)]}")
    plt.scatter(Z, Y, label="orig", color="m")
    plt.plot(Z, Z * H + c, label="lin reg")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    ff = np.fft.rfft(xs)
    freq = lambda e: np.fft.rfftfreq(e)
    ax[0].stem(1 / 365 / freq(len(ff) * 2 - 1), np.abs(ff), markerfmt=" ")
    ax[0].set_title("Fourier")
    _, _, A, _ = prony(len(xs) // 2, xs)
    ax[1].stem(freq(len(xs) - 1), A, markerfmt=" ")
    ax[1].set_title("Prony")
    plt.show()


def task(n=200, h=.02, m=2):
    ps = np.arange(1, n + 1)
    xs = np.array(sum(k * np.exp(-h * ps / k) * np.cos(4 * np.pi * k * h * ps + np.pi / k) for k in range(1, 4)))

    l, o, A, p = prony(n // m, xs)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    ff = np.fft.rfft(xs)
    freq = lambda e: np.fft.rfftfreq(e)
    axs[0].stem(freq(len(ff) * 2 - 1), np.abs(ff), markerfmt=" ")
    axs[0].set_title("Fourier")
    axs[1].stem(freq(len(A) * 2 - 1), A, markerfmt=" ")
    axs[1].set_title("Prony")
    plt.show()
    # # _xs = np.array([np.sum(A*np.exp(-l*k+1j*(o*k+p))) for k in ps])

    # scrapper()
    # print(read_data())
    hurst(read_data())


if __name__ == "__main__":
    task()
