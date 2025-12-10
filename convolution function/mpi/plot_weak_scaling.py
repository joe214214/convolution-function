import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件
try:
    df = pd.read_csv('weak.csv', header=None)
    
    # 检查文件是否为空
    if df.empty:
        print("错误: weak.csv 文件为空!")
        exit(1)
    
    # 2. 根据你的文件格式手动定义列名
    df.columns = [
        'timestamp', 'H', 'W', 'K', 'S', 'P', 'Cin', 'Cout', 'batch',
        'ranks', 'px', 'py', 'halo', 't_total', 't_comp', 't_comm',
        'speedup', 'eff', 'host'
    ]
    
    # 确保 ranks 和 t_total 是数值类型
    df['ranks'] = pd.to_numeric(df['ranks'], errors='coerce')
    df['t_total'] = pd.to_numeric(df['t_total'], errors='coerce')
    
    # 删除无效数据
    df = df.dropna(subset=['ranks', 't_total'])
    
    # 打印调试信息
    print(f"数据行数: {len(df)}")
    print(f"可用的 ranks 值: {sorted(df['ranks'].unique())}")
    print(f"\n前几行数据:")
    print(df[['ranks', 't_total', 't_comp', 't_comm']].head())
    
    # 3. 计算 Weak Scaling Efficiency
    # 弱扩展效率 = T1 / TN
    # 尝试获取 ranks=1 的数据,如果没有则使用最小 ranks 值
    rank1_data = df.loc[df['ranks'] == 1, 't_total']
    
    if len(rank1_data) > 0:
        t1 = rank1_data.values[0]
        print(f"\n使用 ranks=1 的基准时间: {t1:.6f} 秒")
    else:
        # 如果没有 ranks=1 的数据,使用最小 ranks 值作为基准
        min_rank = df['ranks'].min()
        t1 = df.loc[df['ranks'] == min_rank, 't_total'].values[0]
        print(f"\n警告: 没有找到 ranks=1 的数据!")
        print(f"使用最小 ranks={min_rank} 的时间作为基准: {t1:.6f} 秒")
    
    df['weak_efficiency'] = t1 / df['t_total']
    
    # 按 ranks 排序
    df = df.sort_values('ranks')
    
    # 4. 绘制 ranks vs t_total
    plt.figure(figsize=(8, 6))
    plt.plot(df['ranks'], df['t_total'], marker='o', linewidth=2, label='Total Runtime (s)')
    plt.title('Weak Scaling: Ranks vs Total Runtime', fontsize=14)
    plt.xlabel('Number of Processes (ranks)', fontsize=12)
    plt.ylabel('Total Runtime (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 只有当 ranks 值合适时才使用 log scale
    if len(df['ranks']) > 1 and df['ranks'].min() > 0:
        plt.xscale('log', base=2)
        plt.xticks(df['ranks'], df['ranks'].astype(int))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('weak_scaling_runtime.png', dpi=300)
    plt.show()
    
    # 5. 绘制 ranks vs weak_efficiency
    plt.figure(figsize=(8, 6))
    plt.plot(df['ranks'], df['weak_efficiency'], marker='o', linewidth=2, color='orange', label='Weak Scaling Efficiency')
    plt.title('Weak Scaling: Ranks vs Efficiency', fontsize=14)
    plt.xlabel('Number of Processes (ranks)', fontsize=12)
    plt.ylabel('Efficiency (T1 / TN)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if len(df['ranks']) > 1 and df['ranks'].min() > 0:
        plt.xscale('log', base=2)
        plt.xticks(df['ranks'], df['ranks'].astype(int))
    
    plt.ylim(0, 1.05)  # 纵坐标 0~1 匀称比例
    plt.legend()
    plt.tight_layout()
    plt.savefig('weak_scaling_efficiency.png', dpi=300)
    plt.show()
    
    print("\n图表已成功生成!")
    
except FileNotFoundError:
    print("错误: 找不到 weak.csv 文件!")
    print("请确保 weak.csv 文件存在于当前目录中。")
except Exception as e:
    print(f"发生错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
