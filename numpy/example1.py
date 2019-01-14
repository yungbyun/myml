import numpy as np

arr = np.arange(12)
'''
print(arr)
print(arr.shape)
print(arr.ndim)

arr = arr.reshape(2, 6)
print(arr)
print(arr.shape)
print(arr.ndim)
'''

arr = arr.reshape(2, 2, 3)
print(arr)
print(arr.shape)
print(arr.ndim)

print(arr.sum(axis=0))
print(arr.sum(axis=1))
print(arr.sum(axis=2))

'''
arr = arr.reshape(9, 2, 5)
print(arr)
print(arr.shape)
print(arr.ndim)

#print(arr.dtype)
#print(type(arr))
'''
