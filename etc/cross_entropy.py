import numpy as np

answer = [0.0000001, 1, 0.000001, 0.000001]
y = [1, 0, 0, 0]

def CrossEntropy(answer, y):
    entropy =[]
    for i in range(len(answer)):
        if y[i] == 1:
            entropy.append((-np.log(answer[i])))
        else:
            entropy.append((-np.log(1 - answer[i])))
    return entropy

cost = CrossEntropy(answer, y)
print(cost)
