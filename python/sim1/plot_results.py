import os
import numpy as np
import matplotlib.pyplot as plt


def plot_results():
    # 绑定绝对路径读取数据文件
    file_path = os.path.join(os.path.dirname(__file__), 'simulation_data.npz')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件 '{file_path}'，请先运行 simulate.py 进行仿真！")

    data = np.load(file_path)
    t = data['t']
    q = data['q']
    dq = data['dq']
    q_d = data['q_d']
    dq_d = data['dq_d']
    f_true = data['f_true']
    fn = data['fn']

    # 提取元数据并在终端打印
    S_mode = int(data['S']) if 'S' in data else 'Unknown'
    S1_mode = int(data['S1']) if 'S1' in data else 'Unknown'
    m_nodes = int(data['m']) if 'm' in data else 'Unknown'

    print(f"Plotting results for configuration: S={S_mode}, S1={S1_mode}, m={m_nodes}")

    # 图 1：位置与速度跟踪图像
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Tracking Performance (S={S_mode}, S1={S1_mode}, m={m_nodes})", fontsize=14)

    # 子图 1：位置跟踪
    plt.subplot(2, 1, 1)
    plt.plot(t, q_d, 'r', label='ideal position', linewidth=2)
    plt.plot(t, q, 'k:', label='position tracking', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Position tracking')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 子图 2：速度跟踪
    plt.subplot(2, 1, 2)
    plt.plot(t, dq_d, 'r', label='ideal speed', linewidth=2)
    plt.plot(t, dq, 'k:', label='speed tracking', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('Speed tracking')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()

    # 图 2：真实扰动 vs RBF估计扰动对比图
    plt.figure(figsize=(10, 5))
    plt.title(f"Disturbance Estimation Performance (S={S_mode}, S1={S1_mode}, m={m_nodes})", fontsize=14)
    plt.plot(t, f_true, 'r', label='Practical uncertainties', linewidth=2)
    plt.plot(t, fn, 'b', label='Estimation uncertainties', linewidth=2)
    plt.xlabel('time(s)')
    plt.ylabel('f and fn')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_results()