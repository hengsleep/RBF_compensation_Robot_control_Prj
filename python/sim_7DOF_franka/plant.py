"""7自由度机械臂（近似Franka FR3构型）被控对象动力学模型

动力学核心使用递归Newton-Euler算法(RNEA)按当前关节角度q数值化地计算
M(q), C(q,dq)dq, G(q)，而不是写死的封闭解析式——这样才能正确体现多关节
机械臂各关节转轴不共面（第1关节绕竖直轴、其余关节因连杆扭转角alpha不同
而分别绕不同方向的轴）带来的复杂惯量耦合。

DH参数近似取自公开的Franka Panda/FR3运动学资料，用来让关节构型贴近真实
机械臂的非共面轴向配置；连杆质量/质心/惯量张量是本项目自定义的合理拟真
数值，不代表官方精确出厂参数。
"""

import math
import numpy as np

N_JOINTS = 7

# ---- Modified DH 参数：[a_i(m), alpha_i(rad), d_i(m)]，theta_i为关节变量 ----
# MDH 参数近似取自 Franka Emika Panda / FR3
DH_A     = [0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0]
DH_ALPHA = [0.0, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, 0.0]
DH_D     = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]

"""自定义连杆参数：质量(kg)、质心位置(各自连杆坐标系下,m)、
主惯量对角分量(质心处、连杆坐标系下, kg*m^2)。整臂总质量约15kg量级，
沿臂展从基座到末端逐渐减小，是合理拟真值而非官方精确数据
"""
# 各连杆质量 (kg)
LINK_MASS = [3.2, 3.0, 2.6, 2.6, 1.8, 1.4, 0.6]
# 各连杆质心位置 (m)
LINK_COM = [
    (0.0, -0.02, 0.05),
    (0.0, 0.03, 0.00),
    (0.03, 0.00, -0.02),
    (-0.03, 0.03, 0.00),
    (0.0, 0.02, 0.06),
    (0.04, 0.00, 0.00),
    (0.0, 0.0, 0.02),
]
# 主惯量对角分量 (kg*m^2)
LINK_INERTIA_DIAG = [
    (0.020, 0.020, 0.010),
    (0.020, 0.010, 0.020),
    (0.010, 0.010, 0.010),
    (0.010, 0.010, 0.010),
    (0.006, 0.006, 0.003),
    (0.003, 0.003, 0.002),
    (0.001, 0.001, 0.001),
]

GRAVITY = 9.81

"""以下是纯Python（元组+math库）实现的向量运算，用来替代numpy在极小
的3维向量/3x3矩阵上的调用——numpy对这种极小数组的函数分发开销
远大于实际计算量，几十万次调用累积下来非常慢，改用原生Python
元组运算后单次rnea调用速度提升一个数量级以上

NumPy 对于大型矩阵/数组计算有极致的 C 语言加速，
但每次调用 np.array 或 NumPy 函数时，
都有较重的 Python API 函数分发（Overhead）与内存分配开销
"""

def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _scal(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)
def _rotx(a, v):
    c, s = math.cos(a), math.sin(a)
    return (v[0], c * v[1] - s * v[2], s * v[1] + c * v[2])
def _rotx_T(a, v):
    c, s = math.cos(a), math.sin(a)
    return (v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2])
def _rotz(t, v):
    c, s = math.cos(t), math.sin(t)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])
def _rotz_T(t, v):
    c, s = math.cos(t), math.sin(t)
    return (c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2])
def _R_apply(j, theta, v):
    """R_{j-1,j} 作用在向量v（v在frame j下）上，结果转换到frame j-1"""
    return _rotx(DH_ALPHA[j], _rotz(theta, v))
def _RT_apply(j, theta, v):
    """R_{j-1,j}^T 作用在向量v（v在frame j-1下）上，结果转换到frame j"""
    return _rotz_T(theta, _rotx_T(DH_ALPHA[j], v))
def _p_vec(j):
    a = DH_ALPHA[j]
    return (DH_A[j], -DH_D[j] * math.sin(a), DH_D[j] * math.cos(a))

"""整个动力学计算的核心，输入为
当前关节位置 $q$、角速度 $\dot{q}$、角加速度 $\ddot{q}$ 以及基座重力向量 $\mathbf{g}$，
输出为各个关节所需的驱动力矩 $\boldsymbol{\tau} \in \mathbb{R}^7$
"""

def rnea(q, dq, ddq, gravity_vec3):
    """递归Newton-Euler算法（纯Python内核）：给定关节位置/速度/加速度及
    重力向量(基座坐标系下)，返回对应的7维关节力矩向量（numpy数组）。
    """
    z = (0.0, 0.0, 1.0)

    omega = (0.0, 0.0, 0.0)
    omega_dot = (0.0, 0.0, 0.0)
    v_dot = _scal(gravity_vec3, -1.0)  # 重力当作基座虚拟线加速度

    F_list = [None] * N_JOINTS
    Nm_list = [None] * N_JOINTS
    p_list = [_p_vec(j) for j in range(N_JOINTS)]

    for j in range(N_JOINTS):
        theta = q[j]
        omega_prev_in_j = _RT_apply(j, theta, omega)
        omega_new = _add(omega_prev_in_j, _scal(z, dq[j]))
        omega_dot_new = _add(
            _add(_RT_apply(j, theta, omega_dot), _scal(z, ddq[j])),
            _cross(omega_prev_in_j, _scal(z, dq[j])),
        )
        pj = p_list[j]
        v_dot_new = _RT_apply(j, theta, _add(
            _add(_cross(omega_dot, pj), _cross(omega, _cross(omega, pj))), v_dot
        ))

        omega, omega_dot, v_dot = omega_new, omega_dot_new, v_dot_new

        pc = LINK_COM[j]
        v_dot_c = _add(_add(_cross(omega_dot, pc), _cross(omega, _cross(omega, pc))), v_dot)
        F_list[j] = _scal(v_dot_c, LINK_MASS[j])

        Ixx, Iyy, Izz = LINK_INERTIA_DIAG[j]
        I_omega_dot = (Ixx * omega_dot[0], Iyy * omega_dot[1], Izz * omega_dot[2])
        I_omega = (Ixx * omega[0], Iyy * omega[1], Izz * omega[2])
        Nm_list[j] = _add(I_omega_dot, _cross(omega, I_omega))

    f_next = (0.0, 0.0, 0.0)
    n_next = (0.0, 0.0, 0.0)
    tau = [0.0] * N_JOINTS

    for j in range(N_JOINTS - 1, -1, -1):
        if j == N_JOINTS - 1:
            f_next_in_j = (0.0, 0.0, 0.0)
            n_next_in_j = (0.0, 0.0, 0.0)
            p_next = (0.0, 0.0, 0.0)
        else:
            f_next_in_j = _R_apply(j + 1, q[j + 1], f_next)
            n_next_in_j = _R_apply(j + 1, q[j + 1], n_next)
            p_next = p_list[j + 1]

        f = _add(f_next_in_j, F_list[j])
        n = _add(
            _add(Nm_list[j], n_next_in_j),
            _add(_cross(LINK_COM[j], F_list[j]), _cross(p_next, f_next_in_j)),
        )
        tau[j] = n[2]  # n @ z
        f_next, n_next = f, n

    return np.array(tau)


# 名义动力学提取与真实被控对象注入
def nominal_dynamics(q, dq):
    """计算名义动力学的 M0(q)、C0(q,dq)@dq（向量，非矩阵）、G0(q)

    用"单位向量法 Unit Acceleration Method"通过多次调用rnea提取M矩阵各列，
    避免手推复杂的Christoffel符号；
    C(q,dq)dq和G(q)则各自一次rnea调用即可直接得到。
    这个函数同时被plant.py（作为"名义"参考）和controller.py（作为
    控制律里的模型基准）调用，保证控制器和被控对象用的是同一套名义模型。
    """
    zero = np.zeros(N_JOINTS)
    g_vec = np.array([0.0, 0.0, -GRAVITY])

    G0 = rnea(q, zero, zero, g_vec)
    Cdq0 = rnea(q, dq, zero, g_vec) - G0

    M0 = np.zeros((N_JOINTS, N_JOINTS))
    for j in range(N_JOINTS):
        ddq_unit = np.zeros(N_JOINTS)
        ddq_unit[j] = 1.0
        M0[:, j] = rnea(q, zero, ddq_unit, np.zeros(3))

    return M0, Cdq0, G0


# 真实被控对象建模
class SevenLinkPlant:
    """7自由度机械臂真实被控对象：在名义模型基础上叠加20%参数失配 +
    与跟踪误差相关的外部扰动，代表真实系统中未被控制器精确知晓的不确定性。
    """

    def __init__(self):
        self.ddq_prev = np.zeros(N_JOINTS)  # 用上一步加速度打破代数环
        # 每个关节的扰动权重略有不同，避免7个关节的扰动完全一样
        self.dist_weights = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7])

    # 参数失配（Parameter Mismatch）注入
    def derivatives(self, t, q, dq, tau, e, de, dynamics=None):
        M0, Cdq0, G0 = dynamics if dynamics is not None else nominal_dynamics(q, dq)

        # 20%参数失配 Parameter Mismatch，代表真实惯量/科氏力/重力与控制器所用名义模型不完全一致
        d_M = 0.2 * M0
        d_C = 0.2 * Cdq0
        d_G = 0.2 * G0

        d1, d2, d3 = 2.0, 3.0, 6.0
        norm_e = np.linalg.norm(e)
        norm_de = np.linalg.norm(de)
        d_scalar = d1 + d2 * norm_e + d3 * norm_de
        # 关键：把这个人为附加扰动定义在"加速度域"（每个关节大约这么多的
        # 未建模角加速度扰动），再乘以M0转换成力矩量纲。这样M0_inv作用回去
        # 时正好精确抵消回加速度域，不会被M0病态的方向（条件数可达上千）放大
        accel_dist = d_scalar * self.dist_weights # 定义在加速度域的扰动 (rad/s^2)
        d_vec = M0 @ accel_dist # 乘以 M0，转换成力矩域 (N·m)

        M0_inv = np.linalg.inv(M0)
        # 用上一步的加速度计算失配惯量项，避免"f依赖ddq、ddq又依赖f"的代数环
        f_true = M0_inv @ (d_M @ self.ddq_prev + d_C + d_G + d_vec)

        # 根据控制力矩 $\boldsymbol{\tau}$ 和真实总扰动 $\boldsymbol{f}_{\text{true}}$，
        # 计算出系统真正的关节加速度 $\boldsymbol{\ddot{q}}$ 
        ddq = M0_inv @ (tau - Cdq0 - G0) + f_true
        self.ddq_prev = ddq.copy()

        return dq, ddq, f_true
