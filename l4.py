import numpy as np

# 2
x, y = np.arange(-10, 6), np.arange(-5, 11)
# 3
# z = np.sort(np.array([e for e in zip(y,x)]).flatten())
z = np.sort(np.array([*zip(y, x)]).flatten())
# 4
norm = lambda x, p=1, ws=True: (
    lambda ws: np.power(np.sum(ws * np.power(abs(x), p)), 1 / p)
)(np.ones_like(x) if not ws else
  (lambda ws: len(x) == len(ws) * np.all(ws > 0) * np.isclose(np.sum(ws), 1) and ws)(
      np.array(input("Enter weights:\n>>> ").split()).astype(np.float64))
  )
# 6
fac = lambda _: _ < 1 or _ * fac(_ - 1)
# 7
v = np.array(input("Enter vec coeffs:\n>>> ").split()).astype(np.float64)
print(f"min: {min(v)}, max: {max(v)}, sum: {sum(v)}")

print(x, y, z, sep="\n")
print(norm(x, ws=False), norm(y, ws=False))
print(norm(x))
print(fac(4))
