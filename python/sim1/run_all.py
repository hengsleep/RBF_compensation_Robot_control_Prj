import sys
import os

try:
    from simulate import main as run_simulation
    from plot_results import plot_results
except ImportError as e:
    print(f"[错误] 模块导入失败，请确保 run_all.py 与 simulate.py, plot_results.py 在同一目录下！\n详细信息: {e}")
    sys.exit(1)


def main():
    # =========================================================================
    # Simulation Configuration (Modify only these three parameters)
    # =========================================================================
    m_nodes = 19      # Number of RBF hidden nodes: 5 or 19 or 50
    S_mode = 3        # Control mode: 1-Nominal CTC, 2-Exact disturbance compensation, 3-RBF compensation
    S1_mode = 2       # Adaptive law: 1-Standard Lyapunov update, 2-UUB robust update
    # =========================================================================

    print("=" * 60)
    print(f" 步骤 1/2: 正在启动仿真 (配置: m={m_nodes}, S={S_mode}, S1={S1_mode})...")
    print("=" * 60)

    # 1. 调用 simulate.py 并传入参数 (同步阻塞执行)
    try:
        run_simulation(m_nodes=m_nodes, S_mode=S_mode, S1_mode=S1_mode)
    except Exception as e:
        print(f"\n[错误] 仿真过程中出现异常中断: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" 步骤 2/2: 仿真计算完成！自动启动结果可视化...")
    print("=" * 60)

    # 2. 自动画图
    try:
        plot_results()
    except Exception as e:
        print(f"\n[错误] 绘图过程中出现异常: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()