import numpy as np
from scipy.linalg import solve_continuous_lyapunov


class RBFController:
    """
    RBF自适应控制器模块 (对应 simulation_1ctrl.m)
    支持 m=5/19, S=1/2/3, S1=1/2 配置
    """

    def __init__(self, m=5, S=3, S1=2):
        self.m = m
        self.S = S
        self.S1 = S1

    # RBF 隐藏层中心点阵列 c (2 x m)
        if self.m == 5:
            self.c = 0.5 * np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), (2, 1))
        elif self.m == 19:
            self.c = (1.0 / 9.0) * np.tile(np.arange(-9.0, 10.0), (2, 1))
        elif self.m == 50:
            # 针对 50 个节点的均匀分布阵列 (在 -2.5 到 2.5 之间线性生成 50 个中心点)
            nodes_50 = np.linspace(-2.5, 2.5, 50)
            self.c = np.tile(nodes_50, (2, 1))
        else:
            raise ValueError("m 参数必须为 5, 19 或 50")

        # 高斯函数宽度与增益参数
        self.b = 5.0
        alfa = 3.0
        self.kp = alfa**2        # 9.0
        self.kv = 2.0 * alfa     # 6.0
        self.gamma = 200.0       # 自适应学习率
        self.k1 = 0.001          # 鲁棒修正系数
        self.M = 10.0            # 名义惯量参数

        # 求解 Lyapunov 方程: A^T * P + P * A = -Q
        A = np.array([
            [0.0, 1.0],
            [-self.kp, -self.kv]
        ])
        Q = 50.0 * np.eye(2)
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.B = np.array([[0.0], [1.0 / self.M]])

    def get_reference_trajectory(self, t):
        """
        解析计算期望轨迹 q_d, dq_d, ddq_d
        """
        q_d = np.sin(t)
        dq_d = np.cos(t)
        ddq_d = -np.sin(t)
        return q_d, dq_d, ddq_d

    def compute_basis(self, xi):
        """
        计算高斯径向基函数输出向量 h
        """
        xi_col = xi.reshape(2, 1)
        h = np.zeros(self.m)
        for j in range(self.m):
            diff = xi_col - self.c[:, j:j + 1]
            norm_sq = np.sum(diff**2)
            h[j] = np.exp(-norm_sq / (2.0 * self.b**2))
        return h

    def compute_control(self, t, q, dq, theta):
        """
        根据 S 计算控制力矩 tau 与 RBF 估计扰动 fn
        """
        q_d, dq_d, ddq_d = self.get_reference_trajectory(t)
        e = q - q_d
        de = dq - dq_d

        # 名义误差反馈控制力矩 tol1
        tol1 = self.M * (ddq_d - self.kv * de - self.kp * e)

        xi = np.array([[e], [de]])
        h = self.compute_basis(xi)

        # 真实扰动计算 (仅用于 S=2 选项的已知扰动精确补偿)
        f_true = -15.0 * dq - 30.0 * np.sign(dq)

        if self.S == 1:          # 名义控制 (无补偿)
            fn = 0.0
            tol = tol1
        elif self.S == 2:        # 精确已知扰动补偿
            fn = 0.0
            tol2 = -f_true
            tol = tol1 + tol2
        elif self.S == 3:        # RBF 自适应逼近补偿
            fn = float(np.dot(theta, h))
            tol2 = -fn
            tol = tol1 + tol2
        else:
            raise ValueError("S 参数必须为 1, 2 或 3")

        return tol, fn

    def compute_weight_derivatives(self, t, q, dq, theta):
        """
        根据 S1 计算自适应更新律 dtheta/dt
        """
        q_d, dq_d, _ = self.get_reference_trajectory(t)
        e = q - q_d
        de = dq - dq_d
        xi = np.array([[e], [de]])

        h = self.compute_basis(xi)

        scalar_term = float(xi.T @ self.P @ self.B)
        norm_xi = float(np.linalg.norm(xi))

        if self.S1 == 1:         # 基础 Lyapunov 自适应律
            dtheta = self.gamma * h * scalar_term
        elif self.S1 == 2:       # 带 UUB 鲁棒项的自适应律
            dtheta = self.gamma * h * scalar_term + self.k1 * self.gamma * norm_xi * theta
        else:
            raise ValueError("S1 参数必须为 1 或 2")

        return dtheta