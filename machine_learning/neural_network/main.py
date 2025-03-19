import pandas as pd
import numpy as np

import my_neural_network as mynn

# 数据集
with open('iris.csv', 'r') as file_object:
    iris = pd.read_csv(file_object)
M = iris.shape[0]  # 样本个数
iris_species = list(iris['Species'].unique())
num_class = len(iris_species)  # 类数
iris_label = np.zeros((num_class,M))
for m in range(M):
    for c in iris_species:
        if iris.iloc[m, 5] == c:
            iris_label[iris_species.index(c), m] = 1  # one-hot

iris_data = iris.iloc[:, 1:5].values.T  # 存储4维数据
mean = np.mean(iris_data, axis=1)[:, np.newaxis]
std = np.std(iris_data, axis=1)[:, np.newaxis]
iris_data = (iris_data - mean) / std  # z-score标准化
shuffle_ix = np.random.permutation(np.arange(M))  # 随机索引
iris_data = iris_data[:, shuffle_ix]
iris_label = iris_label[:, shuffle_ix]

layer_info = [4, 50, 50, 3]
nn = mynn.NN(layer_info[:])
nn.train(epochs=500, A_0=iris_data, Y=iris_label, lr=0.1)

# 测试准确率
acc_num = 0
for index in range(M):
    predict = np.argmax(nn.test(iris_data[:, index][:, np.newaxis]))
    label = list(iris_label[:,index]).index(1.)
    if predict == label:
        acc_num += 1
print("准确率: ", acc_num / M)

