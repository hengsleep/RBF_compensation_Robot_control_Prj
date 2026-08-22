import sys

try:
    from simulate import main as run_simulation
    from plot_results import plot_results
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


def main():
    # ==========================================
    # 7自由度机械臂参数控制台
    # ==========================================
    m_nodes = 19      # RBF隐层节点数：5 或 19
    S_mode = 1        # 控制器模式 (1-无补偿, 2-精确前馈补偿, 3-RBF补偿)
    S1_mode = 2        # 自适应律 (1-标准, 2-带鲁棒项UUB)
    # ==========================================

    print("=" * 60)
    print(f"正在启动7自由度仿真 (配置: m={m_nodes}, S={S_mode}, S1={S1_mode})...")
    print("=" * 60)

    try:
        run_simulation(m_nodes=m_nodes, S_mode=S_mode, S1_mode=S1_mode)
    except Exception as e:
        print(f"\n仿真中断: {e}")
        sys.exit(1)

    print("\n仿真完成，自动启动可视化...")
    plot_results()


if __name__ == '__main__':
    main()
