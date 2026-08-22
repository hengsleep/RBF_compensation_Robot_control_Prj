import os
import numpy as np
import matplotlib.pyplot as plt


def plot_results():
    # 绑定绝对路径读取数据文件
    file_path = os.path.join(os.path.dirname(__file__), 'sim_data_2link.npz')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件 '{file_path}'，请先运行 simulate.py 进行仿真！")

    d = np.load(file_path)
    t = d['t']

    # 提取当前实验配置参数
    S_mode = int(d['S']) if 'S' in d else 'Unknown'
    S1_mode = int(d['S1']) if 'S1' in d else 'Unknown'
    m_nodes = int(d['m']) if 'm' in d else 'Unknown'

    cfg_str = f"S={S_mode}, S1={S1_mode}, m={m_nodes}"
    print(f"Plotting Two-Link results for configuration: {cfg_str}")

    # =========================================================================
    # 图 1：位置跟踪性能 (对应 simulation_2plot.m 中的 figure(1))
    # =========================================================================
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Position Tracking Performance ({cfg_str})", fontsize=14)

    # 子图 1：连杆 1 位置跟踪
    plt.subplot(2, 1, 1)
    plt.plot(t, d['qd'][:, 0], 'r', label='ideal position for link 1', linewidth=2)
    plt.plot(t, d['q'][:, 0], 'k:', label='position tracking for link 1', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Position tracking for link 1')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 子图 2：连杆 2 位置跟踪
    plt.subplot(2, 1, 2)
    plt.plot(t, d['qd'][:, 1], 'r', label='ideal position for link 2', linewidth=2)
    plt.plot(t, d['q'][:, 1], 'k:', label='position tracking for link 2', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Position tracking for link 2')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()

    # =========================================================================
    # 图 2：速度跟踪性能 (对应 simulation_2plot.m 中的 figure(2))
    # =========================================================================
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Speed Tracking Performance ({cfg_str})", fontsize=14)

    # 子图 1：连杆 1 速度跟踪
    plt.subplot(2, 1, 1)
    plt.plot(t, d['dqd'][:, 0], 'r', label='ideal speed for link 1', linewidth=2)
    plt.plot(t, d['dq'][:, 0], 'k:', label='speed tracking for link 1', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Speed tracking for link 1')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 子图 2：连杆 2 速度跟踪
    plt.subplot(2, 1, 2)
    plt.plot(t, d['dqd'][:, 1], 'r', label='ideal speed for link 2', linewidth=2)
    plt.plot(t, d['dq'][:, 1], 'k:', label='speed tracking for link 2', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Speed tracking for link 2')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()

    # =========================================================================
    # 图 3：控制力矩输入 (对应 simulation_2plot.m 中的 figure(3))
    # =========================================================================
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Control Inputs ({cfg_str})", fontsize=14)

    # 子图 1：连杆 1 控制输入
    plt.subplot(2, 1, 1)
    plt.plot(t, d['tol'][:, 0], 'r', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Control input of link 1')
    plt.grid(True)

    # 子图 2：连杆 2 控制输入
    plt.subplot(2, 1, 2)
    plt.plot(t, d['tol'][:, 1], 'r', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Control input of link 2')
    plt.grid(True)

    plt.tight_layout()

    # =========================================================================
    # 图 4：不确定性扰动 vs RBF 估计对比 (对应 simulation_2plot.m 中的 figure(4))
    # =========================================================================
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Disturbance Estimation Performance ({cfg_str})", fontsize=14)

    # 子图 1：连杆 1 扰动估计 (f1 vs fn1)
    plt.subplot(2, 1, 1)
    plt.plot(t, d['f_true'][:, 0], 'r', label='Practical uncertainties of link 1', linewidth=2)
    plt.plot(t, d['fn'][:, 0], 'k:', label='Estimation uncertainties of link 1', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('f1 and fn1')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 子图 2：连杆 2 扰动估计 (f2 vs fn2)
    plt.subplot(2, 1, 2)
    plt.plot(t, d['f_true'][:, 1], 'r', label='Practical uncertainties of link 2', linewidth=2)
    plt.plot(t, d['fn'][:, 1], 'k:', label='Estimation uncertainties of link 2', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('f2 and fn2')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_results()