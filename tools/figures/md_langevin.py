"""Writes md_langevin.tex: forces on one bead (Newton + friction + noise), the time-stepping, and the mean squared displacement.

Run: python3 md_langevin.py
"""
import numpy as np

from figlib import Axes, Fig

f = Fig()
f.panel("a", "Newton's second law for one bead", 0, 8.4, 13.4, 12.6)
f.panel("b", "Time is cut into small steps $\\Delta t$", 0, 4.0, 13.4, 8.2)
f.panel("c", "How far a bead travels in time $t$", 0, -1.8, 13.4, 3.8)
# ---- a: forces on a bead
f.add("  \\fill[cohesin!55] (3.6,10.3) circle (0.4);\\draw[cohesin,line width=1pt] (3.6,10.3) circle (0.4);\n")
for (dx, dy, t, ang) in ((-1.5, 0.5, "springs, angle, repulsion: $-\\nabla U$", "west"), (1.5, 0.9, "friction: $-\\gamma\\,m\\,\\mathbf v$", "west"), (0.3, -1.0, "random kicks: $\\boldsymbol\\xi(t)$", "west")):
    f.add(f"  \\draw[mv,cohesin,line width=1.6pt] (3.6,10.3)--({3.6 + dx:.2f},{10.3 + dy:.2f});\n")
f.add("  \\node[note,anchor=east] at (2.1,11.2) {$-\\nabla U$};\n")
f.add("  \\node[note,anchor=west] at (5.2,11.3) {$-\\gamma m\\mathbf v$};\n")
f.add("  \\node[note,anchor=west] at (4.1,9.2) {$\\boldsymbol\\xi(t)$};\n")
f.add("  \\node[lab,anchor=west] at (7.4,11.5) {$m\\,\\ddot{\\mathbf r}=-\\nabla U-\\gamma m\\,\\dot{\\mathbf r}+\\boldsymbol\\xi(t)$};\n")
f.add("  \\node[note,text width=5.4cm,anchor=north west] at (7.4,10.9) {mass $\\times$ acceleration = sum of forces: $-\\nabla U$ from the springs, angles and repulsion ($U$ is the potential energy of the chain), friction, and random kicks. Friction and kicks stand for the surrounding fluid and set the temperature.};\n")
# ---- b: stepping
t = np.arange(8); xs = np.cumsum([0, 0.5, 0.2, 0.7, 0.3, -0.1, 0.5, 0.6])
ax = Axes(f, 1.4, 5.0, 5.6, 2.1, (0, 7.3), (-0.3, 2.3), "time", "position $x$", xticks=(), yticks=())
ax.line(t, xs, "line width=1pt,ink!40", 1)
for a, b in zip(t, xs):
    X, Y = ax.p(a, b); f.add(f"  \\fill[cohesin] ({X:.2f},{Y:.2f}) circle (0.1);\n")
X0, Y0 = ax.p(0, -0.3); X1, _ = ax.p(1, -0.3)
f.add(f"  \\draw[ink!60,{{Stealth}}-{{Stealth}}] ({X0:.2f},{Y0 - 0.2:.2f})--({X1:.2f},{Y0 - 0.2:.2f});\\node[note] at ({(X0 + X1) / 2:.2f},{Y0 - 0.4:.2f}) {{$\\Delta t$}};\n")
f.add("  \\node[note,text width=5.0cm,anchor=north west] at (7.8,7.8) {Every step: compute the force on each bead from the current positions, then update velocity and position a little. Repeat thousands of times.\\\\[3pt]$\\Delta t$ = \\texttt{integrator\\_step} (100 fs). Too large a step makes the chain blow up.};\n")
# ---- c: MSD
ts = np.logspace(-1, 2, 60)
msd = np.where(ts < 1, ts ** 2, ts)
ax = Axes(f, 1.6, -0.6, 4.2, 2.6, (-1, 2), (-2, 2), "$\\log_{10}$ time", "$\\log_{10}$ MSD", xticks=(), yticks=())
ax.line(np.log10(ts), np.log10(msd), "cur", 1)
f.add("  \\node[note,anchor=west] at (2.0,1.6) {$\\propto t^2$};\\node[note,anchor=west] at (4.4,0.6) {$\\propto t$};\n")
f.add("  \\node[note,text width=5.8cm,anchor=north west] at (6.8,2.8) {\\textbf{Mean squared displacement} (MSD): the average of $|\\mathbf r(t)-\\mathbf r(0)|^2$. At short times the bead moves freely (ballistic, $t^2$); later friction and kicks make it a random walk (diffusive, $t$). The friction sets when this change happens.};\n")
f.save(__file__)
