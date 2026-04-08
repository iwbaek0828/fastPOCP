import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from matplotlib.colors import ListedColormap

import sys

# =========================
# 1) Load ANI matrix
# =========================
infile = sys.argv[1]
df = pd.read_csv(infile, index_col=0).apply(pd.to_numeric, errors="coerce")

common = df.index.intersection(df.columns)
df = df.loc[common, common]

if df.isna().any().any():
    df = df.fillna(np.nanmedian(df.values))

df = (df + df.T) / 2.0
np.fill_diagonal(df.values, 100.0)

# =========================
# 2) Clustering distance (원본 fastpocp 군집)
# =========================
ani = df.values.astype(float)
dist = 1.0 - (ani / 100.0)
dist = np.clip(dist, 0.0, 1.0)
np.fill_diagonal(dist, 0.0)

Z = linkage(squareform(dist, checks=False), method="average")

# =========================
# 3) 5단위 binning: 값 -> 카테고리(0~6)
#   0: <40 (회색)
#   1: 40-50
#   2: 50-60
#   3: 60-70
#   4: 70-80
#   5: 98-90
#   6: 90-100
# =========================
# 100 포함 안정화를 위해 100.0001
edges = np.array([40, 50, 60, 70, 80, 90, 100.0001], dtype=float)

vals = df.clip(upper=100).values  # >100만 100으로
cats = np.digitize(vals, edges, right=False)  # 0(<70), 1([70,75)), ..., 6([95,100.0001)), 7(>=100.0001)
cats = np.clip(cats, 0, 6)  # 혹시 100.0001 이상이면 마지막 bin으로

plot_df = pd.DataFrame(cats, index=df.index, columns=df.columns)

# =========================
# 4) Discrete colormap (회색 + 6색)
# =========================
base = plt.get_cmap("YlGnBu")  #
bin_colors = base(np.linspace(0.15, 0.95, 6))  # 70~100 구간 6개 색
gray = np.array([[0.78, 0.78, 0.78, 1.0]])     # <40 회색
cmap_discrete = ListedColormap(np.vstack([gray, bin_colors]))

# 컬러바 tick/라벨
ticks = np.arange(0, 7, 1)
ticklabels = ["<40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]

# =========================
# 5) Draw clustermap (※ norm 사용 안 함 → 에러 원천 제거)
# =========================
sns.set_context("notebook")

g = sns.clustermap(
    plot_df,
    row_linkage=Z,
    col_linkage=Z,
    cmap=cmap_discrete,
    vmin=-0.5, vmax=6.5,          # 카테고리 0~6을 딱 고정
    xticklabels=False,
    yticklabels=False,
    figsize=(10, 10),
    linewidths=0.0,
    cbar_kws={
        "ticks": ticks,
        "spacing": "uniform",
        "drawedges": True          # 컬러바 경계선(계단) 강조
    }
)

g.cax.set_yticklabels(ticklabels)
# g.ax_heatmap.set_title("ANI heatmap (<70 gray, 70–100 step=5) + dendrogram", pad=12)

out_png = sys.argv[2]
g.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {out_png}")
