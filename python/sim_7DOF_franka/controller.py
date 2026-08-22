"""7自由度RBF神经网络自适应控制器

沿用1DOF/2DOF版本一致的设计：外环用计算力矩控制律生成虚拟加速度指令，
内环用名义动力学做逆动力学补偿，再叠加RBF网络对未建模不确定性的在线估计。
控制器和被控对象共用同一套名义动力学计算（从plant.py导入），只是被控
对象在此基础上额外叠加了20%参数失配和外部扰动——这部分才是控制器
需要靠RBF在线估计去补偿的"未知"部分。
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from plant import nominal_dynamics, N_JOINTS
from input import get_reference_trajectory


class SevenLinkRBFController:
    def __init__(self, m=5, S=3, S1=2):
        self.n = N_JOINTS
        self.m = m
        self.S = S
        self.S1 = S1

        # 传统全网格法配置高斯基函数的中心点 5^14
        # RBF中心点矩阵：(2n x m)，沿用2DOF的做法——同一组m个标量中心点
        # 沿状态向量每一维铺开，而不是做成 m^(2n) 的全网格（否则节点数会爆炸）
        if self.m == 5:
            base_c = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        else:
            base_c = np.linspace(-2.5, 2.5, self.m)
        self.c = np.tile(base_c, (2 * self.n, 1))

        self.b = 6.0  # 高斯基函数宽度：状态维度从2DOF的4维升到14维后适当调宽，
                       # 避免高维空间里距离过大导致基函数值恒为0（维度灾难）
        alfa = 3.0
        self.kp = (alfa ** 2) * np.eye(self.n)
        self.kv = (2 * alfa) * np.eye(self.n)
        self.gamma = 20.0
        self.k1 = 0.001

        # 求解Lyapunov方程 A'P+PA=-Q，A/B为分块矩阵，维度(2n x 2n)/(2n x n)
        A = np.block([
            [np.zeros((self.n, self.n)), np.eye(self.n)],
            [-self.kp, -self.kv],
        ])
        B = np.block([
            [np.zeros((self.n, self.n))],
            [np.eye(self.n)],
        ])
        Q = 50.0 * np.eye(2 * self.n)
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.B = B

        # S=2（假设扰动精确已知的理想化基准）用的扰动权重，须和plant.py里
        # SevenLinkPlant.dist_weights保持一致，否则S=2就不是真正的"精确补偿"
        self.dist_weights = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7])

    def get_reference_trajectory(self, t):
        return get_reference_trajectory(t)

    def compute_basis(self, xi):
        diff = xi.reshape(-1, 1) - self.c  # (2n, m)
        norm_sq = np.sum(diff ** 2, axis=0)  # (m,)
        return np.exp(-norm_sq / (2.0 * self.b ** 2))

    def compute_control(self, t, q, dq, ddq_current, W, dynamics=None):
        """计算控制力矩tau，并返回扰动诊断信息

        ddq_current: 当前时刻的真实加速度（取自plant.ddq_prev，只有S=2
        这种理想化基准控制器才会用到，代表"假设扰动精确已知"的作弊读数）
        dynamics: 可选，外部预先算好的(M0,Cdq0,G0)，避免和plant.py重复计算
        """
        q_d, dq_d, ddq_d = self.get_reference_trajectory(t)
        e = q - q_d
        de = dq - dq_d
        xi = np.concatenate([e, de])

        M0, Cdq0, G0 = dynamics if dynamics is not None else nominal_dynamics(q, dq)
        v = ddq_d - self.kv @ de - self.kp @ e
        tol1 = M0 @ v + Cdq0 + G0

        if self.S == 1:
            fn = np.zeros(self.n)
            tau = tol1
        elif self.S == 2:
            # 理想化基准：假设20%参数失配与外部扰动都精确已知（现实中不可行）
            d_M = 0.2 * M0
            d_C = 0.2 * Cdq0
            d_G = 0.2 * G0
            d1, d2, d3 = 2.0, 3.0, 6.0
            norm_e = np.linalg.norm(e)
            norm_de = np.linalg.norm(de)
            d_scalar = d1 + d2 * norm_e + d3 * norm_de
            accel_dist = d_scalar * self.dist_weights
            d_vec = M0 @ accel_dist
            M0_inv = np.linalg.inv(M0)
            f_exact = M0_inv @ (d_M @ ddq_current + d_C + d_G + d_vec)
            fn = np.zeros(self.n)
            tau = tol1 - M0 @ f_exact
        elif self.S == 3:
            h = self.compute_basis(xi)
            fn = W @ h
            tau = tol1 - M0 @ fn
        else:
            raise ValueError("S 参数必须为 1, 2 或 3")

        return tau, fn, e, de

    def compute_weight_derivatives(self, t, q, dq, W):
        q_d, dq_d, _ = self.get_reference_trajectory(t)
        e = q - q_d
        de = dq - dq_d
        xi = np.concatenate([e, de])
        h = self.compute_basis(xi)

        scalar_vector = xi @ self.P @ self.B  # (n,)
        dw_base = self.gamma * np.outer(h, scalar_vector).T  # (n, m)

        if self.S1 == 1:
            dw = dw_base
        elif self.S1 == 2:
            norm_W = np.linalg.norm(W.flatten())
            dw = dw_base - self.k1 * self.gamma * norm_W * W
        else:
            raise ValueError("S1 参数必须为 1 或 2")

        return dw
