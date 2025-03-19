import numpy as np
import matplotlib.pyplot as plt

class NN():
    def __init__(self, layer_info):
        self.n = layer_info  # 各层神经元个数
        self.layer_num = len(layer_info) - 1
        self.W = np.empty(self.layer_num + 1, dtype=object)  # 第一层 W 索引为 1, 不考虑W[0]
        self.b = np.empty(self.layer_num + 1, dtype=object)
        # 前向传播
        self.Z = np.empty(self.layer_num + 1, dtype=object)
        self.A = np.empty(self.layer_num + 1, dtype=object)
        # 反向传播
        self.dZ = np.empty(self.layer_num + 1, dtype=object)
        self.dA = np.empty(self.layer_num + 1, dtype=object)
        self.dW = np.empty(self.layer_num + 1, dtype=object)
        self.db = np.empty(self.layer_num + 1, dtype=object)

        self.weight_init()

    def weight_init(self):
        """权重初始化"""
        for l in range(1, self.layer_num + 1):
            self.W[l] = np.random.random((self.n[l], self.n[l-1])) - 0.5  # 随机初始化
            self.b[l] = np.zeros((self.n[l], 1))

    def train(self, epochs, A_0, Y, lr=0.1):
        print('Training...'.center(60, '='))
        loss_list = []
        for epoch in range(epochs):
            # 前向传播
            self.forward_propagation(A_0)
            # 计算loss
            loss_list.append(self.loss(Y))
            # 反向传播
            self.backward_propagation(A_0, Y)
            # 权重更新
            self.weights_update(lr)
            print(f"Epoch:{epoch+1}/{epochs} loss:{loss_list[-1]}")

        plt.plot(list(range(epochs)), loss_list)
        plt.show()

    def test(self, input):
        for l in range(1, self.layer_num + 1):
            if l == 1:
                self.Z[l] = self.W[l].dot(input) + self.b[l]
            else:
                self.Z[l] = self.W[l].dot(self.A[l-1]) + self.b[l]
            if l == self.layer_num:
                self.A[l] = self.softmax(self.Z[l])
            else:
                self.A[l] = self.sigmoid(self.Z[l])
        return self.A[self.layer_num]

    def forward_propagation(self, A_0):
        """前向传播"""
        for l in range(1, self.layer_num + 1):
            if l == 1:
                self.Z[l] = self.W[l].dot(A_0) + self.b[l]
            else:
                self.Z[l] = self.W[l].dot(self.A[l-1]) + self.b[l]
            if l == self.layer_num:
                self.A[l] = self.softmax(self.Z[l])
            else:
                self.A[l] = self.sigmoid(self.Z[l])

    def backward_propagation(self, A_0, Y):
        """反向传播"""
        # Y 为样本标签(n, m)
        m = Y.shape[1]
        for l in range(self.layer_num, 0, -1):
            if l == self.layer_num:
                self.dZ[l] = self.A[l] - Y
            else:
                self.dZ[l] = self.dA[l] * self.derivative_sigmoid(self.Z[l])
            if l == 1:
                self.dW[l] = 1 / m * self.dZ[l].dot(A_0.T)
            else:
                self.dW[l] = 1 / m * self.dZ[l].dot(self.A[l - 1].T)
            self.db[l] = 1 / m * np.sum(self.dZ[l], axis=1, keepdims=True)
            self.dA[l-1] = np.dot(self.W[l].T, self.dZ[l])

    def weights_update(self, lr):
        """梯度更新"""
        for l in range(1, self.layer_num + 1):
            self.W[l] -= lr * self.dW[l]
            self.b[l] -= lr * self.db[l]

    def loss(self, Y):
        loss = 0.
        for m in range(Y.shape[1]):
            for i in range(Y.shape[0]):
                if Y[i, m] == 1.:
                    loss += -np.log(self.A[self.layer_num][i, m])  # 交叉熵损失
        loss = 1 / Y.shape[1] * loss
        return loss

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        y = np.zeros(x.shape)
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                y[r, c] = np.exp(x[r, c])
        exp_sum = np.sum(y, axis=0)
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                y[r, c] = y[r, c] / exp_sum[c]
        return y