"""Writes md_forces.tex: the potentials of the force field (bond, angle, excluded volume, compartment well) and the loop spring.

Run: python3 md_forces.py
"""
import numpy as np

from figlib import Axes, Fig

f = Fig()
f.panel("a", "Bonds and angles keep the chain connected and stiff", 0, 8.6, 13.4, 12.6)
f.panel("b", "Excluded volume repels, a Gaussian well attracts", 0, 4.0, 13.4, 8.4)
f.panel("c", "Loop springs pull the two anchors of a loop together", 0, -0.6, 13.4, 3.8)
# ---- a: harmonic potentials
r = np.linspace(0.6, 1.4, 60)
ax = Axes(f, 1.4, 9.6, 3.4, 2.0, (0.6, 1.4), (0, 1), "bond length $r$", "$U_{\\mathrm{bond}}$", xticks=(1.0,), xfmt="$r_0$", yticks=())
ax.line(r, (r - 1) ** 2 / 0.64, "cur", 1)
th = np.linspace(2.0, 4.28, 60)
ax = Axes(f, 6.3, 9.6, 3.4, 2.0, (2.0, 4.28), (0, 1), "angle $\\theta$", "$U_{\\mathrm{angle}}$", xticks=(np.pi,), xfmt="$\\pi$", yticks=())
ax.line(th, (th - np.pi) ** 2 / 1.3, "cur", 1)
f.add("  \\node[note,text width=3.0cm,anchor=north west] at (10.2,12.0) {$U=\\tfrac12k(r-r_0)^2$\\\\[2pt]$U=\\tfrac12k_\\theta(\\theta-\\pi)^2$\\\\[2pt]minimum: straight chain, bond length $r_0$};\n")
# ---- b: EV and Gaussian well
r = np.linspace(0.02, 1.2, 100)
ax = Axes(f, 1.4, 5.2, 4.4, 2.3, (0, 1.2), (0, 1.1), "distance $r$ (nm)", "$U_{\\mathrm{ev}}$", xticks=(0, 0.4, 0.8, 1.2), yticks=())
for p, col in ((3, "cohesin!40"), (6, "cohesin"), (9, "ink!75")):
    u = (0.5 / (r + 0.1)) ** p; ax.line(r, np.minimum(u / 3.0, 1.1), f"line width=1.7pt,{col}", 1)
f.add("  \\node[note,anchor=west,text=cohesin!60] at (3.4,7.2) {$p=3$};\\node[note,anchor=west,text=cohesin] at (3.4,6.8) {$p=6$ (default)};\\node[note,anchor=west,text=ink!75] at (3.4,6.4) {$p=9$};\n")
ax = Axes(f, 7.1, 5.2, 3.0, 2.3, (0, 3), (-1.1, 0.1), "distance $r$", "$U_{\\mathrm{comp}}$", xticks=(), yticks=())
rr = np.linspace(0, 3, 80); ax.line(rr, -np.exp(-rr ** 2 / 2), "cur", 1)
f.add("  \\node[note,text width=2.5cm,anchor=north west] at (10.5,7.8) {\\textbf{ev}: soft power-law wall, so strands do not cross.\\\\[3pt]\\textbf{well}: same-state beads attract.};\n")
# ---- c: loop spring picture
f.add("  \\begin{scope}[yshift=0.9cm]\n")
f.add("  \\draw[fiber,ink!70,line width=1.8pt] (1.0,0.2) .. controls (2.5,0.5) and (3.0,-0.2) .. (4.0,0.1);\n")
f.add("  \\draw[fiber,ink!70,line width=1.8pt] (4.0,0.1) .. controls (4.8,1.1) and (6.0,1.1) .. (6.8,0.1) .. controls (7.4,-0.3) and (8.2,0.3) .. (9.0,0.1);\n")
f.add("  \\fill[cohesin] (4.0,0.1) circle (0.14);\\fill[cohesin] (6.8,0.1) circle (0.14);\n")
zz = " -- ".join(f"({4.0 + 2.8 * i / 14:.2f},{0.1 + (0.12 if i % 2 else -0.12) * (0 < i < 14):.2f})" for i in range(15))
f.add(f"  \\draw[cohesin,line width=1.4pt] {zz};\n")
f.add("  \\node[note] at (5.4,-0.55) {spring between the anchors};\n")
f.add("  \\node[note,text width=3.4cm,anchor=north west] at (9.5,1.9) {$U=\\tfrac12k_L(r-r_L)^2$ with a stiff $k_L$ and a short $r_L$: the anchors are practically in contact. The springs are replaced in every frame.};\n")
f.add("  \\end{scope}\n")
f.save(__file__)
