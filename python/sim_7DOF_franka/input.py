"""期望轨迹生成模块：7个关节的期望位置/速度/加速度

7条轨迹的幅值、频率、相位、偏置都刻意设置得不一样，避免所有关节做
完全相同的正弦运动（那样测试不出关节间惯量耦合的效果）。全部使用
解析式手动给出一阶、二阶导数，不做数值微分。
"""

import numpy as np

N_JOINTS = 7

# 每个关节: 偏置, 幅值, 角频率, 相位
_OFFSET = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7])
_AMP    = np.array([0.4, 0.3, 0.35, 0.25, 0.3, 0.2, 0.3])
_FREQ   = np.array([0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.45])   # rad/s
_PHASE  = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 0.8])


def get_reference_trajectory(t):
    """返回该时刻7个关节的期望位置/速度/加速度，形状均为 (7,)"""
    arg = _FREQ * t + _PHASE
    # $$q_{d,i}(t) = q_{\text{offset},i} + A_i \sin(\omega_i t + \phi_i)$$
    q_d = _OFFSET + _AMP * np.sin(arg)
    # 手动求导,不做数值微分
    dq_d = _AMP * _FREQ * np.cos(arg)
    ddq_d = -_AMP * (_FREQ ** 2) * np.sin(arg)
    return q_d, dq_d, ddq_d
