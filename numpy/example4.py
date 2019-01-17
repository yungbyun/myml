import numpy as np

x = np.array([1, 2])
print(x)
print(x.shape)

y = np.expand_dims(x, axis=0)
print(y)
print(y.shape)

z = np.expand_dims(y, axis=0)
print(z)
print(z.shape)

z = np.expand_dims(z, axis=3)
print(z)
print(z.shape)


p = np.array([[[1, 2]]])
print(p)
print(p.shape)
print(type(p))

