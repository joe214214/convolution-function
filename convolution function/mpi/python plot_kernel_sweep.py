import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读入你的 sweep 结果 - 移除 header=None,让pandas自动识别标题
df = pd.read_csv('sweep.csv')

# 2. 检查列名是否正确,如果CSV没有标题则手动指定
# 如果CSV文件确实有标题,下面的重命名可以注释掉
# 如果列名不匹配,保留这段代码
expected_columns = [
    'timestamp',  # 时间戳
    'H', 'W',     # 图像尺寸
    'K',          # kernel size
    'S',          # stride
    'P',          # padding (实际是pad大小)
    'Cin', 'Cout',
    'batch',
    'ranks', 'px', 'py', 'halo',
    't_total', 't_comp', 't_comm',
    'speedup', 'eff', 'host'
]

# 只有当CSV列数匹配但列名不对时才重命名
if len(df.columns) == len(expected_columns):
    df.columns = expected_columns

# 3. 确保我们用到的列类型正确,添加错误处理
try:
    df['H'] = pd.to_numeric(df['H'], errors='coerce').astype(int)
    df['W'] = pd.to_numeric(df['W'], errors='coerce').astype(int)
    df['K'] = pd.to_numeric(df['K'], errors='coerce').astype(int)
    df['t_total'] = pd.to_numeric(df['t_total'], errors='coerce').astype(float)
    
    # 删除可能存在的无效行
    df = df.dropna(subset=['H', 'W', 'K', 't_total'])
except Exception as e:
    print(f"数据类型转换错误: {e}")
    print("前5行数据:")
    print(df.head())
    raise

# 4. 我们关心三种分辨率：1024、2048、4096
sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
size_labels = {
    (1024, 1024): '1024 x 1024',
    (2048, 2048): '2048 x 2048',
    (4096, 4096): '4096 x 4096',
}

# 为了稳定，按照 kernel size 排序
curves = []
for (h, w) in sizes:
    sub = df[(df['H'] == h) & (df['W'] == w)].copy()
    if len(sub) == 0:
        print(f"警告: 没有找到分辨率 {h}x{w} 的数据")
        continue
    sub = sub.sort_values('K')  # 排序 K=3,7,11
    curves.append((size_labels[(h, w)], sub['K'].values, sub['t_total'].values))

if not curves:
    print("错误: 没有找到任何符合条件的数据进行绘图")
    exit(1)

# 5. 开始画图
plt.figure(figsize=(7,5))

for label, ks, tts in curves:
    plt.plot(
        ks,
        tts,
        marker='o',
        linewidth=2,
        label=label
    )

# y轴设从0开始，保持比例直观
plt.ylim(0, max(df['t_total']) * 1.05)

# x轴就用 kernel size
plt.xticks(sorted(df['K'].unique()))

plt.xlabel('Kernel Size (K x K)', fontsize=12)
plt.ylabel('Total Runtime (seconds)', fontsize=12)
plt.title('Runtime vs Kernel Size at Different Input Resolutions (8 MPI processes)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Input Resolution')
plt.tight_layout()

plt.savefig('kernel_sweep_runtime.png', dpi=300)
plt.show()

print("✅ Saved kernel_sweep_runtime.png")
