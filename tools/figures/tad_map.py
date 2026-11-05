"""Cartoon Hi-C map for the TAD figure: decay along the diagonal, two domains, corner loop dots, a stripe."""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

N, T = 120, 60                                           # bins, TAD boundary
i, j = np.meshgrid(np.arange(N), np.arange(N))
d = np.abs(i - j) + 1.0
same = (i < T) == (j < T)
M = np.where(same, 1.0, 0.22) * d ** -0.75
M = M / M.max()
for a, b in [(3, 54), (66, 117), (12, 40), (74, 102)]:   # loop dots (symmetric)
    for p, q in ((a, b), (b, a)):
        M += 0.30 * np.exp(-((i - p) ** 2 + (j - q) ** 2) / 6)
M[:T, 0] += 0.03; M[0, :T] += 0.03                      # a faint stripe along the first anchor
M = gaussian_filter(M, 0.8)
cmap = LinearSegmentedColormap.from_list("ls", ["#ffffff", "#e6defc", "#8f6cf0", "#4a24c4"])
fig = plt.figure(figsize=(4, 4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(np.clip(M, 0, 1) ** 0.6, cmap=cmap, interpolation="bilinear")
ax.axis("off")
fig.savefig(__file__.replace("tad_map.py", "tad_map.png"), facecolor="white")
