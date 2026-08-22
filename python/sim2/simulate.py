import os
import numpy as np
from controller import TwoLinkRBFController
from plant import TwoLinkPlant

def main(m_nodes=5, S_mode=3, S1_mode=2):
    t0, tf, dt = 0.0, 10.0, 0.001
    t_steps = int((tf - t0) / dt) + 1
    t_array = np.linspace(t0, tf, t_steps)

    ctrl = TwoLinkRBFController(m=m_nodes, S=S_mode, S1=S1_mode)
    plant = TwoLinkPlant()

    # 状态向量 X: [q1, q2, dq1, dq2, W_flat(2*m)]
    state_dim = 4 + 2 * m_nodes
    X = np.zeros(state_dim)
    # 根据 simulation_2plant.m 设置初始状态
    X[0:2] = [0.6, 0.5]  # q1, q2
    X[2:4] = [0.3, 0.5]  # dq1, dq2
    # 权重初始值 0.1
    X[4:] = 0.1 

    log_data = {'t': np.zeros(t_steps), 'q': np.zeros((t_steps, 2)), 
                'dq': np.zeros((t_steps, 2)), 'qd': np.zeros((t_steps, 2)), 
                'dqd': np.zeros((t_steps, 2)), 'tol': np.zeros((t_steps, 2)), 
                'f_true': np.zeros((t_steps, 2)), 'fn': np.zeros((t_steps, 2))}

    def system_derivatives(t_curr, state):
        q = state[0:2].reshape(2, 1)
        dq = state[2:4].reshape(2, 1)
        W = state[4:].reshape(2, m_nodes)

        # 传入 plant.ddq_prev 作为当前的实际加速度 ddq
        ddq_curr = plant.ddq_prev
        tol, fn, e, de = ctrl.compute_control(t_curr, q, dq, ddq_curr, W, plant)
        dW = ctrl.compute_weight_derivatives(t_curr, q, dq, W)
        _, ddq, f_true = plant.derivatives(t_curr, q, dq, tol, e, de)

        dX = np.zeros(state_dim)
        dX[0:2] = dq.flatten()
        dX[2:4] = ddq.flatten()
        dX[4:] = dW.flatten()
        return dX, tol, fn, f_true

    print(f"Starting Two-Link simulation (m={m_nodes}, S={S_mode}, S1={S1_mode})...")
    for i in range(t_steps):
        t_curr = t_array[i]
        
        qd, dqd, _ = ctrl.get_reference_trajectory(t_curr)
        dX, tol, fn, f_true = system_derivatives(t_curr, X)

        log_data['t'][i] = t_curr
        log_data['q'][i, :] = X[0:2]
        log_data['dq'][i, :] = X[2:4]
        log_data['qd'][i, :] = qd.flatten()
        log_data['dqd'][i, :] = dqd.flatten()
        log_data['tol'][i, :] = tol.flatten()
        log_data['f_true'][i, :] = f_true.flatten()
        log_data['fn'][i, :] = fn.flatten()

        if i < t_steps - 1:
            k1 = dX
            k2, _, _, _ = system_derivatives(t_curr + 0.5 * dt, X + 0.5 * dt * k1)
            k3, _, _, _ = system_derivatives(t_curr + 0.5 * dt, X + 0.5 * dt * k2)
            k4, _, _, _ = system_derivatives(t_curr + dt, X + dt * k3)
            X = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    save_path = os.path.join(os.path.dirname(__file__), 'sim_data_2link.npz')
    np.savez(save_path, **log_data, S=S_mode, S1=S1_mode, m=m_nodes)
    print(f"Simulation finished. Data saved to '{save_path}'.")

if __name__ == '__main__':
    main()