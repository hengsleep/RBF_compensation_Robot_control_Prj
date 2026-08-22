import numpy as np
from scipy.linalg import solve_continuous_lyapunov


class TwoLinkRBFController:
    """
    双连杆 RBF 神经网络控制器 (对应 simulation_2ctrl.m 和 simulation_2input.m)
    """

    def __init__(self, m=5, S=3, S1=2):
        self.m = m
        self.S = S
        self.S1 = S1

        # RBF 中心点阵列 (4 x m)
        if self.m == 5:
            base_c = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            self.c = np.tile(base_c, (4, 1))
        else:
            base_c = np.linspace(-2.5, 2.5, self.m)
            self.c = np.tile(base_c, (4, 1))

        self.b = 3.0
        alfa = 3.0
        self.kp = np.array([[alfa**2, 0.0], [0.0, alfa**2]])
        self.kv = np.array([[2 * alfa, 0.0], [0.0, 2 * alfa]])
        self.gamma = 20.0
        self.k1 = 0.001

        # 求解 Lyapunov 方程: A^T * P + P * A = -Q
        A = np.block([
            [np.zeros((2, 2)), np.eye(2)],
            [-self.kp, -self.kv]
        ])
        B = np.block([
            [np.zeros((2, 2))],
            [np.eye(2)]
        ])
        Q = 50.0 * np.eye(4)
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.B = B

    def get_reference_trajectory(self, t):
        qd1 = 1.0 + 0.2 * np.sin(0.5 * np.pi * t)
        qd2 = 1.0 - 0.2 * np.cos(0.5 * np.pi * t)
        dqd1 = 0.2 * 0.5 * np.pi * np.cos(0.5 * np.pi * t)
        dqd2 = 0.2 * 0.5 * np.pi * np.sin(0.5 * np.pi * t)
        ddqd1 = -0.2 * (0.5 * np.pi)**2 * np.sin(0.5 * np.pi * t)
        ddqd2 = 0.2 * (0.5 * np.pi)**2 * np.cos(0.5 * np.pi * t)

        qd = np.array([[qd1], [qd2]])
        dqd = np.array([[dqd1], [dqd2]])
        ddqd = np.array([[ddqd1], [ddqd2]])
        return qd, dqd, ddqd

    def compute_basis(self, xi):
        h = np.zeros((self.m, 1))
        for j in range(self.m):
            diff = xi - self.c[:, j:j + 1]
            norm_sq = np.sum(diff**2)
            h[j, 0] = np.exp(-norm_sq / (2.0 * self.b**2))
        return h

    def compute_control(self, t, q, dq, ddq, W, plant_obj):
        """
        计算控制力矩 tol 与扰动估计 fn
        :param ddq: 系统的实际加速度向量 (2 x 1)，用于 S=2 时解析计算真实扰动 f
        """
        qd, dqd, ddqd = self.get_reference_trajectory(t)
        e = q - qd
        de = dq - dqd
        xi = np.vstack((e, de))
        h = self.compute_basis(xi)

        q1, q2 = q[0, 0], q[1, 0]
        dq1, dq2 = dq[0, 0], dq[1, 0]

        # 名义动力学矩阵 M0, C0, G0
        M0 = np.array([
            [plant_obj.v + plant_obj.p1 + 2 * plant_obj.p2 * np.cos(q2), plant_obj.p1 + plant_obj.p2 * np.cos(q2)],
            [plant_obj.p1 + plant_obj.p2 * np.cos(q2), plant_obj.p1]
        ])
        C0 = np.array([
            [-plant_obj.p2 * dq2 * np.sin(q2), -plant_obj.p2 * (dq1 + dq2) * np.sin(q2)],
            [plant_obj.p2 * dq1 * np.sin(q2), 0.0]
        ])
        G0 = np.array([
            [15 * plant_obj.g * np.cos(q1) + 8.75 * plant_obj.g * np.cos(q1 + q2)],
            [8.75 * plant_obj.g * np.cos(q1 + q2)]
        ])

        # 基础名义控制力矩 tol1
        tol1 = M0 @ (ddqd - self.kv @ de - self.kp @ e) + C0 @ dq + G0

        if self.S == 1:
            # 模式 1：名义控制 (无扰动补偿)
            fn = np.zeros((2, 1))
            tol = tol1

        elif self.S == 2:
            # 模式 2：精确已知扰动前馈补偿 (Modified Computed Torque)
            d_M = 0.2 * M0
            d_C = 0.2 * C0
            d_G = 0.2 * G0

            d1, d2, d3 = 2.0, 3.0, 6.0
            norm_e = np.linalg.norm([e[0, 0], e[1, 0]])
            norm_de = np.linalg.norm([de[0, 0], de[1, 0]])
            d_scalar = d1 + d2 * norm_e + d3 * norm_de
            d_vec = np.array([[d_scalar], [d_scalar]])

            # 根据实际加速度 ddq 计算真实集总不确定性 f
            f_exact = np.linalg.inv(M0) @ (d_M @ ddq + d_C @ dq + d_G + d_vec)

            fn = np.zeros((2, 1))  # S=2 不是 RBF 神经网络估算，故估计输出 fn 为 0
            tol2 = -M0 @ f_exact
            tol = tol1 + tol2

        elif self.S == 3:
            # 模式 3：RBF 神经网络自适应补偿
            fn = W @ h
            tol2 = -M0 @ fn
            tol = tol1 + tol2

        else:
            raise ValueError("S 参数必须为 1, 2 或 3")

        return tol, fn, e, de

    def compute_weight_derivatives(self, t, q, dq, W):
        qd, dqd, _ = self.get_reference_trajectory(t)
        e = q - qd
        de = dq - dqd
        xi = np.vstack((e, de))
        h = self.compute_basis(xi)

        # 基础 Lyapunov 自适应项: \gamma (h \xi^T P B)^T
        scalar_vector = xi.T @ self.P @ self.B        # 维度: (1 x 2)
        dw_base = (self.gamma * h @ scalar_vector).T   # 维度: (2 x m)

        if self.S1 == 1:
            # 模式 1：标准 Lyapunov 自适应律
            dw = dw_base

        elif self.S1 == 2:
            # 模式 2：带 UUB 鲁棒泄漏项的自适应律
            norm_W = np.linalg.norm(W.flatten())
            dw = dw_base - self.k1 * self.gamma * norm_W * W

        else:
            raise ValueError("S1 参数必须为 1 或 2")

        return dw