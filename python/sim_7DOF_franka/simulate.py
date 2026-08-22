"""7自由度仿真主循环：耦合controller.py与plant.py，手写RK4积分
（不依赖scipy.integrate，风格与2DOF版本一致）
"""
import os
import numpy as np
from controller import SevenLinkRBFController
from plant import SevenLinkPlant, N_JOINTS, nominal_dynamics
from input import get_reference_trajectory, _OFFSET


def main(m_nodes=5, S_mode=3, S1_mode=2):
    t0, tf, dt = 0.0, 10.0, 0.001
    t_steps = int((tf - t0) / dt) + 1
    t_array = np.linspace(t0, tf, t_steps)
    n = N_JOINTS

    ctrl = SevenLinkRBFController(m=m_nodes, S=S_mode, S1=S1_mode)
    plant = SevenLinkPlant()

    # 状态向量 X: [q(7), dq(7), W_flat(7*m)]
    # 将系统的所有动态变量统一压平拼接为一个一维数组
    state_dim = 2 * n + n * m_nodes
    X = np.zeros(state_dim)
    X[0:n] = _OFFSET + 0.3          # 初始位置：在期望轨迹偏置基础上人为加0.3rad误差
    X[n:2 * n] = np.zeros(n)        # 初始速度为0
    X[2 * n:] = 0.1                 # 权值初值，和2DOF版本一致设为0.1

    log_data = {
        't': np.zeros(t_steps), 'q': np.zeros((t_steps, n)),
        'dq': np.zeros((t_steps, n)), 'qd': np.zeros((t_steps, n)),
        'dqd': np.zeros((t_steps, n)), 'tol': np.zeros((t_steps, n)),
        'f_true': np.zeros((t_steps, n)), 'fn': np.zeros((t_steps, n)),
    }

    def system_derivatives(t_curr, state):
        q = state[0:n]
        dq = state[n:2 * n]
        W = state[2 * n:].reshape(n, m_nodes)

        ddq_curr = plant.ddq_prev
        dynamics = nominal_dynamics(q, dq)
        tau, fn, e, de = ctrl.compute_control(t_curr, q, dq, ddq_curr, W, dynamics=dynamics)
        dW = ctrl.compute_weight_derivatives(t_curr, q, dq, W)
        _, ddq, f_true = plant.derivatives(t_curr, q, dq, tau, e, de, dynamics=dynamics)

        dX = np.zeros(state_dim)
        dX[0:n] = dq
        dX[n:2 * n] = ddq
        dX[2 * n:] = dW.flatten()
        return dX, tau, fn, f_true

    print(f"Starting 7-DOF simulation (m={m_nodes}, S={S_mode}, S1={S1_mode})...")
    for i in range(t_steps):
        t_curr = t_array[i]

        # 四阶龙格-库塔（RK4）积分器, Runge-Kutta 数值积分算法
        q_d, dq_d, _ = get_reference_trajectory(t_curr)
        dX, tau, fn, f_true = system_derivatives(t_curr, X)

        log_data['t'][i] = t_curr
        log_data['q'][i, :] = X[0:n]
        log_data['dq'][i, :] = X[n:2 * n]
        log_data['qd'][i, :] = q_d
        log_data['dqd'][i, :] = dq_d
        log_data['tol'][i, :] = tau
        log_data['f_true'][i, :] = f_true
        log_data['fn'][i, :] = fn

        if i < t_steps - 1:
            k1 = dX # k2, k3, k4 的试探计算只是数值积分内部的临时中间量（Virtual Probes）
            k2, _, _, _ = system_derivatives(t_curr + 0.5 * dt, X + 0.5 * dt * k1)
            k3, _, _, _ = system_derivatives(t_curr + 0.5 * dt, X + 0.5 * dt * k2)
            k4, _, _, _ = system_derivatives(t_curr + dt, X + dt * k3)
            X = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if i % 2000 == 0:
            print(f"  t={t_curr:.2f}s  |e|={np.linalg.norm(X[0:n]-q_d):.4f}")

    save_path = os.path.join(os.path.dirname(__file__), 'sim_data_7dof.npz')
    np.savez(save_path, **log_data, S=S_mode, S1=S1_mode, m=m_nodes)
    print(f"Simulation finished. Data saved to '{save_path}'.")


if __name__ == '__main__':
    main()
