import numpy as np

class TwoLinkPlant:
    """
    双连杆机械臂动力学模型 (对应 simulation_2plant.m)
    """
    def __init__(self):
        # 系统参数
        self.v = 13.33
        self.p1 = 8.98
        self.p2 = 8.75
        self.g = 9.8
        
        # 初始状态
        self.ddq_prev = np.zeros((2, 1))

    def derivatives(self, t, q, dq, tol, e, de):
        q1, q2 = q[0, 0], q[1, 0]
        dq1, dq2 = dq[0, 0], dq[1, 0]
        e1, e2 = e[0, 0], e[1, 0]
        de1, de2 = de[0, 0], de[1, 0]

        # 标称动力学矩阵 M0, C0, G0
        M0 = np.array([
            [self.v + self.p1 + 2 * self.p2 * np.cos(q2), self.p1 + self.p2 * np.cos(q2)],
            [self.p1 + self.p2 * np.cos(q2),              self.p1]
        ])
        C0 = np.array([
            [-self.p2 * dq2 * np.sin(q2), -self.p2 * (dq1 + dq2) * np.sin(q2)],
            [ self.p2 * dq1 * np.sin(q2),  0.0]
        ])
        G0 = np.array([
            [15 * self.g * np.cos(q1) + 8.75 * self.g * np.cos(q1 + q2)],
            [8.75 * self.g * np.cos(q1 + q2)]
        ])

        # 不确定性参数
        d_M = 0.2 * M0
        d_C = 0.2 * C0
        d_G = 0.2 * G0

        # 外加扰动
        d1, d2, d3 = 2.0, 3.0, 6.0
        norm_e = np.linalg.norm([e1, e2])
        norm_de = np.linalg.norm([de1, de2])
        d_scalar = d1 + d2 * norm_e + d3 * norm_de
        d = np.array([[d_scalar], [d_scalar]])

        # 真实集总扰动 f (使用上一步的加速度打破代数环)
        M0_inv = np.linalg.inv(M0)
        f = M0_inv @ (d_M @ self.ddq_prev + d_C @ dq + d_G + d)

        # 计算实际加速度 ddq
        ddq = M0_inv @ (tol - C0 @ dq - G0) + f
        
        # 更新用于下一步代数环的加速度
        self.ddq_prev = ddq.copy()

        return dq, ddq, f