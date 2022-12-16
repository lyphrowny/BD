import numpy as np
import matplotlib.pyplot as plt
from l9 import moving_median, turns, kandell
from l10 import moving_exponent


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
    *ys, = map(lambda y: (y,), range(2000, 2023))
    ms = [range(11, 13), *[range(1, 13)] * (len(ys) - 2), range(1, 12)]
    for (m, y) in chain.from_iterable(starmap(product, zip(ms, ys))):
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


def weather(xs, *, fmt=.6):
    alpha = 0.03
    w_size = 101
    me = moving_exponent(xs, alpha)
    mm = moving_median(xs, w_size)
    for d, lab in zip((xs, me, mm),
                      f"orig mov_exp({alpha:.2f}) mov_mean({w_size})".split()):
        plt.plot(d, label=lab)
    plt.legend()
    plt.show()

    for name, trend in zip(("moving exponent", "moving median"), (me, mm)):
        detrend = xs - trend
        p, e_p = turns(detrend)
        print(f"\t{name}:\n"
              f"\t\tp: {p:{fmt}f}, E(p): {e_p:{fmt}f}\n"
              f"\t\tKandell: {kandell(detrend):{fmt}f}")

    ff = np.fft.rfft(xs)
    freq = lambda e: np.fft.rfftfreq(e)
    plt.stem(freq(len(ff) * 2 - 1) / 365, np.abs(ff), markerfmt=" ")
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

    # scrapper()
    # print(read_data())
    weather(read_data())


if __name__ == "__main__":
    # scrapper()
    task()
