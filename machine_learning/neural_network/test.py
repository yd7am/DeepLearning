import numpy as np

def softmax(x):
    y = np.zeros(x.shape)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            y[r, c] = np.exp(x[r, c])
    exp_sum = np.sum(y, axis=0)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            y[r, c] = y[r, c] / exp_sum[c]
    return y

a = np.array([[1,2,3],[1,5,6],[1,8,9]])
b = softmax(a)
print(b)