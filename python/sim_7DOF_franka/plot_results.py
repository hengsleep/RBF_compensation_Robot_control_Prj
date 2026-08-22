import os
import numpy as np
import matplotlib.pyplot as plt

N_JOINTS = 7


def plot_results():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'sim_data_7dof.npz')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件 '{file_path}'，请先运行 simulate.py 进行仿真！")

    # 创建保存图片的文件夹 result plot（如果不存在）
    output_dir = os.path.join(script_dir, 'result plot')
    os.makedirs(output_dir, exist_ok=True)

    d = np.load(file_path)
    t = d['t']

    S_mode = int(d['S']) if 'S' in d else 'Unknown'
    S1_mode = int(d['S1']) if 'S1' in d else 'Unknown'
    m_nodes = int(d['m']) if 'm' in d else 'Unknown'
    cfg_str = f"S={S_mode}, S1={S1_mode}, m={m_nodes}"
    print(f"Plotting 7-DOF results for configuration: {cfg_str}")

    # 图1：7个关节的位置跟踪，4x2排布（最后一格留空）
    fig1, axes = plt.subplots(4, 2, figsize=(13, 14))
    fig1.suptitle(f"Position Tracking Performance ({cfg_str})", fontsize=14)
    axes = axes.flatten()
    for j in range(N_JOINTS):
        ax = axes[j]
        ax.plot(t, d['qd'][:, j], 'r', label='ideal', linewidth=1.8)
        ax.plot(t, d['q'][:, j], 'k:', label='actual', linewidth=1.8)
        ax.set_xlabel('time(s)')
        ax.set_ylabel(f'Joint {j+1} position')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)
    axes[-1].axis('off')
    fig1.tight_layout()

    # 图2：7个关节的速度跟踪
    fig2, axes2 = plt.subplots(4, 2, figsize=(13, 14))
    fig2.suptitle(f"Speed Tracking Performance ({cfg_str})", fontsize=14)
    axes2 = axes2.flatten()
    for j in range(N_JOINTS):
        ax = axes2[j]
        ax.plot(t, d['dqd'][:, j], 'r', label='ideal', linewidth=1.8)
        ax.plot(t, d['dq'][:, j], 'k:', label='actual', linewidth=1.8)
        ax.set_xlabel('time(s)')
        ax.set_ylabel(f'Joint {j+1} speed')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)
    axes2[-1].axis('off')
    fig2.tight_layout()

    # 图3：7个关节的控制力矩输入
    fig3, axes3 = plt.subplots(4, 2, figsize=(13, 14))
    fig3.suptitle(f"Control Inputs ({cfg_str})", fontsize=14)
    axes3 = axes3.flatten()
    for j in range(N_JOINTS):
        ax = axes3[j]
        ax.plot(t, d['tol'][:, j], 'b', linewidth=1.5)
        ax.set_xlabel('time(s)')
        ax.set_ylabel(f'Joint {j+1} torque')
        ax.grid(True)
    axes3[-1].axis('off')
    fig3.tight_layout()

    # 图4：7个关节的真实扰动 vs RBF估计扰动对比
    fig4, axes4 = plt.subplots(4, 2, figsize=(13, 14))
    fig4.suptitle(f"Disturbance Estimation Performance ({cfg_str})", fontsize=14)
    axes4 = axes4.flatten()
    for j in range(N_JOINTS):
        ax = axes4[j]
        ax.plot(t, d['f_true'][:, j], 'r', label='true', linewidth=1.8)
        ax.plot(t, d['fn'][:, j], 'k:', label='estimated', linewidth=1.8)
        ax.set_xlabel('time(s)')
        ax.set_ylabel(f'Joint {j+1} disturbance')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)
    axes4[-1].axis('off')
    fig4.tight_layout()

    # 保存图片至 result plot 目录下
    fig1.savefig(os.path.join(output_dir, 'tracking_position.png'), dpi=130)
    fig2.savefig(os.path.join(output_dir, 'tracking_speed.png'), dpi=130)
    fig3.savefig(os.path.join(output_dir, 'control_inputs.png'), dpi=130)
    fig4.savefig(os.path.join(output_dir, 'disturbance_estimation.png'), dpi=130)
    print(f"图像已保存至目录: {output_dir}")

    plt.show()


if __name__ == '__main__':
    plot_results()