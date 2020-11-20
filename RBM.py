# 实现受限玻尔兹曼机，暂仅考虑可视层、隐藏神经元取值均为二进制的情况
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.bias_a = np.zeros(self.n_visible)  # 可视层偏移量
        self.bias_b = np.zeros(self.n_hidden)  # 隐藏层偏移量
        self.weights = np.random.normal(0, 0.01, size=(self.n_visible, self.n_hidden))
        self.n_sample = None

    def encode(self, v):
        # 编码，即基于v计算h的条件概率：p(h=1|v)
        return sigmoid(self.bias_b + np.dot(v, self.weights))

    def decode(self, h):
        # 解码(重构)：即基于h计算v的条件概率：p(v=1|h)
        return sigmoid(self.bias_a + h @ self.weights.T)

    def gibbs_sample(self, v0, max_cd):
        # gibbs采样, 返回max_cd采样后的v以及h值
        v = v0
        for _ in range(max_cd):
            # 首先根据输入样本对每个隐藏层神经元采样。二项分布采样，决定神经元是否激活
            ph = self.encode(v)
            h = np.random.binomial(1, ph, (self.n_sample, self.n_hidden))
            # 根据采样后隐藏层神经元取值对每个可视层神经元采样
            pv = self.decode(h)
            v = np.random.binomial(1, pv, (self.n_sample, self.n_visible))
        return v

    def update(self, v0, v_cd, eta):
        # 根据Gibbs采样得到的可视层取值(解码或重构)，更新参数
        ph = self.encode(v0)
        ph_cd = self.encode(v_cd)
        self.weights += eta * (v0.T @ ph - v_cd.T @ ph)  # 更新连接权重参数
        self.bias_b += eta * np.mean(ph - ph_cd, axis=0)  # 更新隐藏层偏移量b
        self.bias_a += eta * np.mean(v0 - v_cd, axis=0)  # 更新可视层偏移量a
        return

    def fit(self, data, max_step=100, max_cd=2, eta=0.1):
        """
        训练主函数,采用对比散度算法(CD算法)更新参数
        :param data: 训练数据集, (n_sample, n_input)
        :param max_step: 最大迭代步数
        :param max_cd: 采样步数
        :param eta: 学习率
        :return:
        """
        assert data.shape[1] == self.n_visible, "输入数据维度与可视层神经元数目不相等"
        self.n_sample = data.shape[0]

        for i in range(max_step):
            v_cd = self.gibbs_sample(data, max_cd)
            self.update(data, v_cd, eta)
            error = np.sum((data - v_cd) ** 2) / self.n_sample / self.n_visible * 100
            if not i % 100:  # 将重构后的样本与原始样本对比计算误差
                print("可视层状态误差比例:{0}%".format(round(error, 2)))
        return

    def predict(self, v):
        # 输入训练数据，预测隐藏层输出
        ph = self.encode(v)[0]
        states = ph >= np.random.rand(len(ph))
        return states.astype(int)


if __name__ == '__main__':
    rbm_model = RBM(n_visible=6, n_hidden=2)
    train_data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]]) 
    rbm_model.fit(train_data, max_step=1000, max_cd=1, eta=0.1)
    print(rbm_model.weights, rbm_model.bias_a, rbm_model.bias_b)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(rbm_model.predict(user))
"""
测试数据引用自参考资料5。该数据的含义：
每个样本对应一个用户对6部电影的评分，简化为0(不好看)和1(好看)，
6部电影分别属于奥斯卡获奖影片和奇幻影片，对应两个潜在因子，即2个隐藏层神经元，
据此可以判定用户的电影喜好类别。
"""