"""Cartoon Hi-C map with a stripe: one leg rests at a CTCF site, the other sweeps along the row; averaged over many loadings."""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

N, A, B = 120, 12, 100                                   # bins, fixed anchor (forward CTCF), far boundary (reverse CTCF)
rng = np.random.default_rng(3)
i, j = np.meshgrid(np.arange(N), np.arange(N))
M = (np.abs(i - j) + 1.0) ** -0.75
M = np.where(((i >= A) & (i <= B)) == ((j >= A) & (j <= B)), M, 0.35 * M)       # weaker contacts across the domain edges
S = np.zeros((N, N))
for _ in range(4000):                                    # one-sided extrusion: the right leg moves until it stops or the ring unloads
    life = rng.exponential(55)
    x = min(A + 1 + int(rng.uniform(0, 2)), B)
    for _ in range(int(min(life, B - A))):
        S[A, x] += 1                                     # contact (A, x) while the loop has this size
        x += 1
        if x >= B:
            S[A, B] += 6                                 # the leg that reaches the reverse CTCF stays there
            break
S = S + S.T
M = M / M.max() * 0.8
S[A, B] = 1.2 * S[A, A + 1:B].max()                     # bright end where the leg stalls at the reverse CTCF
M += 0.55 * gaussian_filter(S, 1.0) / gaussian_filter(S, 1.0).max()
M = gaussian_filter(M, 0.6)
cmap = LinearSegmentedColormap.from_list("ls", ["#ffffff", "#e6defc", "#8f6cf0", "#4a24c4"])
fig = plt.figure(figsize=(4, 4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(np.clip(M, 0, 1) ** 0.6, cmap=cmap, interpolation="bilinear")
ax.axis("off")
fig.savefig(__file__.replace("stripe_map.py", "stripe_map.png"), facecolor="white")
