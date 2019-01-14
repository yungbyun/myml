import numpy as np

x = np.array(([1, 2], [3, 4]))
print(x)

print('\n')
y = np.expand_dims(x, axis=3)  # 새로운 축을 만들
print(y)

'''
[[1 2] [3 4]]
[[[1 2] [3 4]]]
[[[1][2]][[3][4]]]
'''
