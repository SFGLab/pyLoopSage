"""Two linked 3D loops seen from three directions; writes link_views.tex (TikZ pieces) from the real geometry.

Crossings, their signs and Lk come from the same Gauss formula as the program (src/knots/linking.py).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from knots.linking import gauss_linking   # noqa: E402

N = 48
t = np.linspace(0, 2 * np.pi, N, endpoint=False)
wob = lambda k, ph: 1 + 0.12 * np.sin(k * t + ph)                                  # a little thermal wobble
A = np.c_[np.cos(t) * wob(3, 0.4), np.sin(t) * wob(2, 1.1), 0.12 * np.sin(2 * t)]                          # loop 1, roughly in the xy plane
B = np.c_[1.0 + np.cos(t) * wob(2, 2.0), 0.1 * np.sin(3 * t), np.sin(t) * wob(3, 0.9)]                      # loop 2, roughly in the xz plane
B = B[::-1]                                                                           # orient loop 2 so that Lk = +1
LK = gauss_linking(A, B)


def basis(v):
    """Orthonormal screen axes (u, w) for a view along v (v points to the viewer)."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    u = np.cross([0, 0, 1], v)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0, 0])
    return v, u, np.cross(v, u)


def crossings(v):
    """Crossings between loop 1 and loop 2 in the view along v: (i, j, over, sign, point on screen)."""
    v, u, w = basis(v)
    P = lambda X: np.c_[X @ u, X @ w]
    pa, pb = P(A), P(B)
    da, db = A @ v, B @ v
    out = []
    for i in range(N):
        a0, a1 = pa[i], pa[(i + 1) % N]
        for j in range(N):
            b0, b1 = pb[j], pb[(j + 1) % N]
            M = np.array([a1 - a0, b0 - b1]).T
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            s, q = np.linalg.solve(M, b0 - a0)
            if 0 <= s < 1 and 0 <= q < 1:
                ha, hb = da[i] + s * (da[(i + 1) % N] - da[i]), db[j] + q * (db[(j + 1) % N] - db[j])
                over = 0 if ha > hb else 1
                d1, d2 = (A[(i + 1) % N] - A[i]), (B[(j + 1) % N] - B[j])
                top, bot = (d1, d2) if over == 0 else (d2, d1)
                sign = int(np.sign(np.cross(np.r_[top @ u, top @ w, 0], np.r_[bot @ u, bot @ w, 0])[2])) * (1 if over == 0 else 1)
                out.append((i, j, over, sign, a0 + s * (a1 - a0)))
    return out


def panel(v):
    """TikZ code of one view: both loops, over-strands redrawn with a white halo."""
    _, u, w = basis(v)
    raw = np.vstack([np.c_[A @ u, A @ w], np.c_[B @ u, B @ w]])
    lo, hi = raw.min(0), raw.max(0)
    scale = min(5.2 / (hi[0] - lo[0]), 4.4 / (hi[1] - lo[1]))                          # fit the panel
    mid = (lo + hi) / 2
    P = lambda X: (np.c_[X @ u, X @ w] - mid) * scale
    pa, pb = P(A), P(B)
    path = lambda p: " -- ".join(f"({x:.3f},{y:.3f})" for x, y in p) + " -- cycle"
    code = [rf"\draw[lpA] {path(pa)};", rf"\draw[lpB] {path(pb)};"]
    cr = crossings(v)
    for i, j, over, sign, _ in cr:
        X, idx = (pa, i) if over == 0 else (pb, j)
        seg = [X[(idx + k) % N] for k in range(-3, 5)]
        pth = " -- ".join(f"({x:.3f},{y:.3f})" for x, y in seg)
        code += [rf"\draw[halo2] {pth};", rf"\draw[{'lpA' if over == 0 else 'lpB'}] {pth};"]
    for *_, sign, pt in cr:                                                        # sign of each crossing
        x, y = (pt - mid) * scale
        code.append(rf"\node[sgn] at ({x:.3f},{y:.3f}) {{${'+' if sign > 0 else '-'}$}};")
    return "\n".join(code), cr, pa, pb


views = {"a": (0.3, -0.5, 1.0), "b": (0.5, -0.5, 0.6), "c": (0.0, 1.0, 0.0)}
out, report = [], {}
for k, v in views.items():
    code, cr, _, _ = panel(v)
    signs = [c[3] for c in cr]
    report[k] = (len(cr), sum(signs) / 2)
    out.append(rf"\newcommand{{\cap{k}}}{{{len(cr)} crossings: ${r',\ '.join(f'{x:+d}' for x in signs)}$}}")
    out.append(rf"\newcommand{{\view{k}}}{{{code}}}")
Path(__file__).with_name("link_views.tex").write_text("\n".join(out) + "\n")
print("Lk (Gauss):", round(LK, 3), " views (crossings, sum of signs/2):", report)
