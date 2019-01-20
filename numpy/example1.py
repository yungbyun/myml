import numpy as np

arr = np.arange(12)

'''
print(arr)
print(arr.shape)
print(arr.ndim)
print(type(arr))
'''

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
print(arr)
print(arr.shape)
print(arr.ndim)
print(type(arr))

arr2 = np.array([[1,2,3], [4,5,6]])
print(arr2.shape)
print(arr2[0, 0], arr2[0, 1], arr2[1, 0])

a = np.zeros((2, 2))
print(a)

b = np.ones((1, 2))
print(b)

c = np.full((2, 2), 7)
print(c)

d = np.eye(2)
print(d)

e = np.random.random((2, 2))
print(e)


'''
arr = arr.reshape(2, 6)
print(arr)
print(arr.shape)
print(arr.ndim)
'''

'''
arr = arr.reshape(2, 2, 3)
print(arr)
print(arr.shape)
print(arr.ndim)

print(arr.sum(axis=0))
print(arr.sum(axis=1))
print(arr.sum(axis=2))


arr = arr.reshape(9, 2, 5)
print(arr)
print(arr.shape)
print(arr.ndim)

#print(arr.dtype)
#print(type(arr))
'''
