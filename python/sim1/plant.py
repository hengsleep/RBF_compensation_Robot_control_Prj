import numpy as np


class Plant:
    """
    被控物理对象模块 (对应 chap6_1plant.m)
    """

    def __init__(self):
        self.M = 10.0  # 系统真实惯量

    def get_true_disturbance_for_plotting(self, dq):
        """
        上帝视角/诊断接口：计算真实非线性摩擦扰动 d。
        仅用于画图对比，绝不介入控制器的决策链路。
        """
        return -15.0 * dq - 30.0 * np.sign(dq)

    def derivatives(self, t, q, dq, tol):
        """
        根据牛顿第二定律计算物理系统的状态导数 [dq, ddq]
        """
        d = self.get_true_disturbance_for_plotting(dq)
        ddq = (tol + d) / self.M
        return dq, ddq