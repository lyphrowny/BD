import numpy as np
import matplotlib.pyplot as plt


def moving_window(xs, width):
    # to print the window widths use:
    # y = [min(width, len(xs)-1-k, k) for k in range(len(xs))]
    # y = [min(width, len(xs)-k, k) for k in range(1,len(xs))]
    # print(y)
    # print(len(y))
    y = [((m := min(width, len(xs) - 1 - k, k)) or 1) and np.sum(xs[k - m:k + m + 1]) / (2 * m + 1) for k in
         range(len(xs))]
    # y = [(m:=min(width, len(xs)-k, k)) and np.sum(xs[k-m:k+m+1])/(2*m+1) for k in range(1,len(xs))]
    # print(y)
    return np.array(y)


def moving_median(xs, width):
    return np.array(
        [((m := min(width, len(xs) - 1 - k, k)) or 1) and np.median(xs[k - m:k + m + 1]) for k in range(len(xs))])
    # return np.array([(m:=min(width, len(xs)-k, k)) and np.median(xs[k-m:k+m+1]) for k in range(1,len(xs))])


def turns(xs):
    # print(xs[0], xs[1], xs[~1], xs[~0])
    p = np.sum(np.logical_or(np.logical_and(xs[:~1] < xs[1:~0], xs[1:~0] > xs[2:]),
                             np.logical_and(xs[:~1] > xs[1:~0], xs[1:~0] < xs[2:])))
    return p, 2 / 3 * (len(xs) - 2)


def kandell(xs):
    p = np.sum([len(np.where(xs[i] < xs[i + 1:])[0]) for i in range(len(xs) - 1)])
    # print(xs[:10])
    # print(len(np.where(xs[2]>xs[3:])[0]))
    # print(p)
    # p = np.sum(xs[:~0]>xs[1:])
    # print(p)
    n = len(xs)
    # print(n*(n-1))
    return 4 * p / (n * (n - 1)) - 1


def task(ws, h=.05, dist_size=1000, fmt=.6):
    exact = np.sqrt(h * np.arange(1000))
    y = exact + np.random.normal(0, 1, dist_size)

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), tight_layout=True)

    for w, ax in zip(ws, axs):
        mw = moving_window(y, w)
        mm = moving_median(y, w)

        ax.plot(exact, label="ex")
        ax.plot(mw, label="mw")
        ax.plot(mm, label="mm")
        ax.legend()
        ax.set_title(f"window: {2 * w + 1}")
        # ax.show()

        print(f"window: {2 * w + 1}")
        for name, trend in zip(("moving average", "moving median"), (mw, mm)):
            detrend = y - trend
            p, e_p = turns(detrend)
            print(f"\t{name}:\n"
                  f"\t\tp: {p:{fmt}f}, E(p): {e_p:{fmt}f}\n"
                  f"\t\tKandell: {kandell(detrend):{fmt}f}")
        print()
        # dmw = y - mw
        # dmm = y - mm
        # print(turns(y))
        # print(kandell(y))

    plt.show()


if __name__ == "__main__":
    task([10, 20, 55])
