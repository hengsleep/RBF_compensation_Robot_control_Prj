import os
import numpy as np
from controller import RBFController
from plant import Plant


def main(m_nodes=5, S_mode=3, S1_mode=2):
    """
    仿真主函数，接收从外部传入的三个配置参数
    :param m_nodes: 神经网络节点数 (5 或 19)
    :param S_mode:  控制器模式 (1-CTC名义控制, 2-已知扰动精确补偿, 3-RBF补偿)
    :param S1_mode: 自适应律模式 (1-基础Lyapunov律, 2-带UUB鲁棒项)
    """
    # 仿真时间设置
    t0 = 0.0
    tf = 10.0
    dt = 0.001
    t_steps = int((tf - t0) / dt) + 1
    t_array = np.linspace(t0, tf, t_steps)

    # 实例化组件
    ctrl = RBFController(m=m_nodes, S=S_mode, S1=S1_mode)
    plant = Plant()

    # 动态构建连续状态向量 X = [q, dq, theta_1, ..., theta_m]^T
    state_dim = 2 + m_nodes
    X = np.zeros(state_dim)
    X[0] = 0.6  # 初始位置 q0
    X[1] = 0.0  # 初始速度 dq0

    # 数据记录数组
    log_t = np.zeros(t_steps)
    log_q = np.zeros(t_steps)
    log_dq = np.zeros(t_steps)
    log_q_d = np.zeros(t_steps)
    log_dq_d = np.zeros(t_steps)
    log_tol = np.zeros(t_steps)
    log_f_true = np.zeros(t_steps)
    log_fn = np.zeros(t_steps)

    def system_derivatives(t_curr, state):
        q = state[0]
        dq = state[1]
        theta = state[2:]

        tol, _ = ctrl.compute_control(t_curr, q, dq, theta)
        dtheta = ctrl.compute_weight_derivatives(t_curr, q, dq, theta)
        _, ddq = plant.derivatives(t_curr, q, dq, tol)

        dX = np.zeros(state_dim)
        dX[0] = dq
        dX[1] = ddq
        dX[2:] = dtheta
        return dX

    # 仿真主循环 (RK4 积分器)
    print(f"Starting simulation (m={m_nodes}, S={S_mode}, S1={S1_mode})...")
    for i in range(t_steps):
        t_curr = t_array[i]
        q_curr = X[0]
        dq_curr = X[1]
        theta_curr = X[2:]

        q_d, dq_d, _ = ctrl.get_reference_trajectory(t_curr)
        tol, fn = ctrl.compute_control(t_curr, q_curr, dq_curr, theta_curr)
        f_true = plant.get_true_disturbance_for_plotting(dq_curr)

        log_t[i] = t_curr
        log_q[i] = q_curr
        log_dq[i] = dq_curr
        log_q_d[i] = q_d
        log_dq_d[i] = dq_d
        log_tol[i] = tol
        log_f_true[i] = f_true
        log_fn[i] = fn

        if i < t_steps - 1:
            k1 = system_derivatives(t_curr, X)
            k2 = system_derivatives(t_curr + 0.5 * dt, X + 0.5 * dt * k1)
            k3 = system_derivatives(t_curr + 0.5 * dt, X + 0.5 * dt * k2)
            k4 = system_derivatives(t_curr + dt, X + dt * k3)

            X = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # 绑定绝对路径保存
    save_path = os.path.join(os.path.dirname(__file__), 'simulation_data.npz')
    np.savez(save_path,
             t=log_t,
             q=log_q,
             dq=log_dq,
             q_d=log_q_d,
             dq_d=log_dq_d,
             tol=log_tol,
             f_true=log_f_true,
             fn=log_fn,
             S=S_mode,
             S1=S1_mode,
             m=m_nodes)
    print(f"Simulation finished. Data saved to '{save_path}'.")


if __name__ == '__main__':
    main()