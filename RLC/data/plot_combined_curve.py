import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import csv

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# ── 从 CSV 读取数据 ────────────────────────────────────────────────────────────
def load_csv(filepath):
    data = {'R=10': {'f': [], 'ur': []}, 'R=30': {'f': [], 'ur': []}}
    with open(filepath, newline='', encoding='utf-8') as f:
        first = f.readline()          # 跳过首行 "frequency_response_data"
        if not first.strip().startswith('f_Hz'):
            pass                      # 确认已跳过，DictReader 读取真正的表头
        reader = csv.DictReader(f)
        for row in reader:
            label = 'R=10' if 'R=10' in row['Experiment'] else 'R=30'
            data[label]['f'].append(float(row['f_Hz']))
            data[label]['ur'].append(float(row['Ur_mV']))
    for key in data:
        data[key]['f']  = np.array(data[key]['f'])
        data[key]['ur'] = np.array(data[key]['ur'])
    return data

data = load_csv('frequency_response_data2.csv.csv')

freq_10 = data['R=10']['f']
ur_10   = data['R=10']['ur']
freq_30 = data['R=30']['f']
ur_30   = data['R=30']['ur']

current_10 = ur_10 / 10.0   # mA
current_30 = ur_30 / 30.0   # mA

# ── PCHIP 单调三次 Hermite 插值（复用自 plot_r30_curve.py）────────────────────
def pchip_interp(x, y, xi):
    """Simple monotone cubic interpolation (Fritsch-Carlson)."""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            m[k] = 0.0
    yi = np.empty_like(xi, dtype=float)
    for i, t in enumerate(xi):
        idx = np.searchsorted(x, t, side='right') - 1
        idx = np.clip(idx, 0, n - 2)
        dx = t - x[idx]
        h_k = h[idx]
        t_ = dx / h_k
        h00 =  2*t_**3 - 3*t_**2 + 1
        h10 =    t_**3 - 2*t_**2 + t_
        h01 = -2*t_**3 + 3*t_**2
        h11 =    t_**3 -   t_**2
        yi[i] = h00*y[idx] + h10*h_k*m[idx] + h01*y[idx+1] + h11*h_k*m[idx+1]
    return yi

f_smooth_10 = np.linspace(freq_10.min(), freq_10.max(), 800)
I_smooth_10 = pchip_interp(freq_10, current_10, f_smooth_10)

f_smooth_30 = np.linspace(freq_30.min(), freq_30.max(), 800)
I_smooth_30 = pchip_interp(freq_30, current_30, f_smooth_30)

# ── 谐振点 ────────────────────────────────────────────────────────────────────
def find_peak(f_smooth, I_smooth):
    idx_peak = np.argmax(I_smooth)
    return f_smooth[idx_peak], I_smooth[idx_peak]

f0_10, I0_10 = find_peak(f_smooth_10, I_smooth_10)
f0_30, I0_30 = find_peak(f_smooth_30, I_smooth_30)

# ── 半功率带宽 ────────────────────────────────────────────────────────────────
def find_bandwidth(f_arr, y_arr, y_half):
    above = y_arr >= y_half
    trans = np.where(np.diff(above.astype(int)))[0]
    if len(trans) >= 2:
        fl = np.interp(y_half,
                       [y_arr[trans[0]], y_arr[trans[0]+1]],
                       [f_arr[trans[0]], f_arr[trans[0]+1]])
        fr = np.interp(y_half,
                       [y_arr[trans[1]+1], y_arr[trans[1]]],
                       [f_arr[trans[1]+1], f_arr[trans[1]]])
        return fl, fr
    return None, None

fl_10, fr_10 = find_bandwidth(f_smooth_10, I_smooth_10, I0_10 / np.sqrt(2))
fl_30, fr_30 = find_bandwidth(f_smooth_30, I_smooth_30, I0_30 / np.sqrt(2))

# ── 绘图 ──────────────────────────────────────────────────────────────────────
COLOR_10 = '#2563EB'   # 蓝色
COLOR_30 = '#DC2626'   # 红色

fig, ax = plt.subplots(figsize=(11, 7.2))

# 拟合曲线
ax.plot(f_smooth_10, I_smooth_10, color=COLOR_10, linewidth=2,
        label=r'$I$-$f$ 曲线（$R=10\,\Omega$）')
ax.plot(f_smooth_30, I_smooth_30, color=COLOR_30, linewidth=2,
        label=r'$I$-$f$ 曲线（$R=30\,\Omega$）')

# 散点
ax.scatter(freq_10, current_10, color=COLOR_10, zorder=5, s=40, alpha=0.85,
           label=r'测量点（$R=10\,\Omega$）')
ax.scatter(freq_30, current_30, color=COLOR_30, zorder=5, s=40, alpha=0.85,
           marker='s', label=r'测量点（$R=30\,\Omega$）')

# 谐振频率辅助线（仅虚线，不加文字标注）
for f0, I0, color in [(f0_10, I0_10, COLOR_10), (f0_30, I0_30, COLOR_30)]:
    ax.axvline(f0, color=color, linestyle='--', linewidth=1.0, alpha=0.5)
    ax.axhline(I0, color=color, linestyle=':',  linewidth=1.0, alpha=0.5)

ax.set_xlabel('频率 $f$ (Hz)', fontsize=13)
ax.set_ylabel('电流 $I$ (mA)', fontsize=13)
ax.set_title(r'RLC 串联电路幅频特性曲线 $I(f)$（$R=10\,\Omega$ 与 $R=30\,\Omega$ 对比）',
             fontsize=14, fontweight='bold', pad=14)
ax.set_xlim(240, 710)
ax.set_ylim(0)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(fontsize=9.5, loc='lower right')

# ── 图下方数据说明 ─────────────────────────────────────────────────────────────
bw_10 = (fr_10 - fl_10) if (fl_10 and fr_10) else float('nan')
bw_30 = (fr_30 - fl_30) if (fl_30 and fr_30) else float('nan')

caption = (
    r'$R=10\,\Omega$：'
    f'$f_{{0}}={f0_10:.0f}$ Hz，$I_{{0}}={I0_10:.2f}$ mA，'
    f'$\\Delta f={bw_10:.1f}$ Hz'
    r'          '
    r'$R=30\,\Omega$：'
    f'$f_{{0}}={f0_30:.0f}$ Hz，$I_{{0}}={I0_30:.2f}$ mA，'
    f'$\\Delta f={bw_30:.1f}$ Hz'
)
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=10.5,
         color='#374151',
         bbox=dict(boxstyle='round,pad=0.4', fc='#F9FAFB', ec='#D1D5DB', alpha=0.9))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('RLC_combined_I_f_curve.png', dpi=180, bbox_inches='tight')
print("图像已保存为 RLC_combined_I_f_curve.png")
