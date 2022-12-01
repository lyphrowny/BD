import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# def moving_window(xs, width):
#     # to print the window widths use:
#     # y = [min(width, len(xs)-1-k, k) for k in range(len(xs))]
#     return np.array([((m:=min(width, len(xs)-1-k, k)) or 1) and np.sum(xs[k-m:k+m+1])/(2*m+1) for k in range(len(xs))])


def moving_exponent(xs, a):
    ys = xs.copy()
    for i, x in enumerate(a * xs[1:], start=1):
        ys[i] = x + (1 - a) * ys[i - 1]
    return ys


# def moving_median(xs, width):
#     return np.array([((m:=min(width, len(xs)-1-k, k)) or 1) and np.median(xs[k-m:k+m+1]) for k in range(len(xs))])


def turns(xs):
    p = np.sum(np.logical_or(np.logical_and(xs[:~1] < xs[1:~0], xs[1:~0] > xs[2:]),
                             np.logical_and(xs[:~1] > xs[1:~0], xs[1:~0] < xs[2:])))
    return p, 2 / 3 * (len(xs) - 2)


def kandell(xs):
    p = np.sum([len(np.where(xs[i] < xs[i + 1:])[0]) for i in range(len(xs) - 1)])
    n = len(xs)
    return 4 * p / (n * (n - 1)) - 1


def task(alphas, h=.1, dist_size=1000, fmt=.6):
    exact = .5 * np.sin(h * np.arange(1000))
    y = exact + np.random.normal(0, 1, dist_size)

    # plt.magnitude_spectrum(y, Fs=1)
    rf = np.fft.rfft(y)
    # plt.plot(np.argmax(rf), np.max(rf), "o", color="red")
    freq = np.fft.rfftfreq(len(y))
    # m = rf.max()
    plt.plot(freq, np.absolute(rf) ** 2)
    # print(np.argmax(np.absolute(rf)**2), np.max(np.absolute(rf)**2))
    plt.plot(np.argmax(np.absolute(rf) ** 2) / (2 * len(rf)), np.max(np.absolute(rf) ** 2), "o", color="red")
    # f_m = np.absolute(np.fft.rfft(y)).max()
    # print(f"main freq: {f_m:{fmt}f}")

    fig, axs = plt.subplots(len(alphas), 1, figsize=(9, 8), tight_layout=True)
    for a, ax in zip(alphas, axs):
        me = moving_exponent(y, a)
        ax.plot(exact, label="ex")
        ax.plot(me, label="me")
        ax.legend()
        ax.set_title(f"coeff: {a}")

        dt = y - me
        n = len(dt)
        sm = np.sum(dt) / n

        print(f"coeff: {a}")
        for name, trend in zip(("moving exponent",), (me,)):
            p, e_p = turns(dt)
            # norm = st.normaltest(dt)[1]
            sh = st.shapiro(dt).pvalue
            print(f"\t{name}:\n"
                  f"\t\tp: {p:{fmt}f}, E(p): {e_p:{fmt}f}\n"
                  f"\t\tKandell: {kandell(dt):{fmt}f}\n"
                  f"\t\tsample mean: {sm:{fmt}f}\n"
                  # f"\t\tsample variance: {np.sum((dt-sm)**2)/(n-1):{fmt}f}")
                  # f"\t\t{norm:{fmt}f} -> {'do not '*(norm<5e-2)}reject hypothesis\n"
                  f"\t\tShapiro-Wilk p_value: {sh:{fmt}f} -> {'do not ' * (sh < 5e-2)}reject hypothesis")
        print()
    plt.show()


if __name__ == "__main__":
    task([.01, .05, .1, .3])
