import pandas as pd
import matplotlib.pyplot as plt

# 读取 summary 数据
df = pd.read_csv('results_summary.csv')

# 取出 ranks, speedup, efficiency
ranks = df['ranks'].astype(int)
speedup = df['speedup_calc'].astype(float)
eff = df['efficiency_calc'].astype(float)

# 绘制加速比曲线
plt.figure(figsize=(6,4))
plt.plot(ranks, speedup, marker='o', label='Speedup')
plt.plot(ranks, ranks, '--', color='gray', label='Ideal linear')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.title('Strong Scaling - Speedup vs Processes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup.png", dpi=300)
plt.show()

# 绘制效率曲线
plt.figure(figsize=(6,4))
plt.plot(ranks, eff, marker='o', color='orange', label='Efficiency')
plt.xlabel('Number of Processes')
plt.ylabel('Efficiency')
plt.title('Strong Scaling - Efficiency vs Processes')
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency.png", dpi=300)
plt.show()
