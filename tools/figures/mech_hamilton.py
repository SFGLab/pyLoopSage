"""Writes mech_hamilton.tex: phase space of an oscillator, energy landscape with Boltzmann weights, and the ensembles.

Run: python3 mech_hamilton.py
"""
import numpy as np

from figlib import Axes, Fig

f = Fig()
f.panel("a", "Phase space: the state of a mechanical system is a point $(q,p)$", 0, 8.8, 13.4, 13.9)
f.panel("b", "From an energy function to probabilities: the Boltzmann law", 0, 3.7, 13.4, 8.5)
f.panel("c", "Ensembles: what is held fixed", 0, -1.4, 13.4, 3.7)
# ---- a: oscillator phase portrait
ax = Axes(f, 1.6, 10.0, 4.2, 3.0, (-1.2, 1.2), (-1.2, 1.2), "position $q$", "momentum $p$")
th = np.linspace(0, 2 * np.pi, 80)
for r in (0.35, 0.7, 1.05):
    ax.line(r * np.cos(th), r * np.sin(th), "line width=1.2pt,cohesin!%d" % (35 + 60 * r), 1)
for r in (0.7,):
    for a in (0.2, np.pi + 0.2):
        X, Y = ax.p(r * np.cos(a), -r * np.sin(a))
        X2, Y2 = ax.p(r * np.cos(a + 0.25), -r * np.sin(a + 0.25))
        f.add(f"  \\draw[cohesin,line width=1.2pt,-{{Stealth[length=5pt]}}] ({X:.2f},{Y:.2f})--({X2:.2f},{Y2:.2f});\n")
ax.label(0, -1.2 - 0.0, "", "note")
f.add("  \\node[note,text width=6.2cm,anchor=north west] at (6.9,13.0) {A system with $N$ particles in 3D has $6N$ numbers: all positions and all momenta. The collection of all of them is the \\textbf{phase space}. Time evolution is a curve in it.};\n")
f.add("  \\node[lab,anchor=west] at (6.9,11.2) {$\\dot q=\\dfrac{\\partial H}{\\partial p},\\qquad \\dot p=-\\dfrac{\\partial H}{\\partial q}$};\n")
f.add("  \\node[note,text width=6.2cm,anchor=north west] at (6.9,10.5) {Hamilton's equations. The loops are curves of constant energy $H$: a closed orbit for the oscillator $H=\\tfrac12p^2+\\tfrac12q^2$.};\n")
# ---- b: landscape and Boltzmann
x = np.linspace(-1.45, 1.45, 120)
E = (x ** 2 - 1) ** 2 * 1.0
axE = Axes(f, 1.6, 5.1, 3.5, 2.6, (-1.45, 1.45), (0, 2.2), "state $x$", "energy $E(x)$")
axE.line(x, E, "cur", 2)
for T, col, name in ((0.12, "B", "low $T$"), (0.6, "A", "high $T$")):
    pass
axP = Axes(f, 6.4, 5.1, 3.5, 2.6, (-1.45, 1.45), (0, 1.1), "state $x$", "probability (scaled)")
for T, col in ((0.12, "cohesin"), (0.6, "ink!70")):
    p = np.exp(-E / T); p /= p.max()
    axP.line(x, p, f"line width=1.8pt,{col}", 2)
f.add("  \\node[note,text width=2.8cm,anchor=north west] at (10.3,7.6) {$P(x)\\propto e^{-E(x)/T}$\\\\[3pt]low energy = likely.\\\\[3pt]\\textcolor{cohesin}{purple}: low $T$\\\\black: high $T$, more energy allowed};\n")
# ---- c: ensembles
boxes = [("microcanonical", "$E,\\,N$ fixed", "isolated"), ("canonical", "$T,\\,N$ fixed", "heat bath"), ("grand canonical", "$T,\\,\\mu$ fixed", "heat and particle bath")]
for k, (name, fixed, how) in enumerate(boxes):
    x0 = 0.8 + 4.2 * k
    wall = "ink,line width=2pt" if k == 0 else ("ink!60,line width=1pt,dashed" if k == 1 else "ink!60,line width=1pt,dotted")
    f.add(f"  \\draw[rounded corners=4pt,{wall},fill=white] ({x0},0.4) rectangle ({x0 + 2.0},2.2);\n")
    for i in range(5):
        f.add(f"  \\fill[cohesin!{50 + 10 * (i % 3)}] ({x0 + 0.4 + 0.3 * i:.2f},{0.9 + 0.25 * ((i * 7) % 5) / 2:.2f}) circle (0.1);\n")
    if k >= 1:
        f.add(f"  \\draw[mv] ({x0 + 2.0},1.3)--({x0 + 2.55},1.3);\\node[note] at ({x0 + 2.95},1.3) {{bath}};\n")
    f.add(f"  \\node[lab] at ({x0 + 1.0},2.6) {{\\bfseries {name}}};\\node[note] at ({x0 + 1.0},-0.1) {{{fixed}}};\n")
f.add("  \\node[box,draw=cohesin!80,fill=cohesin!6,font=\\footnotesize] at (8.7,-0.75) {LoopSage: canonical ($N_{\\mathrm{lef}}$ fixed, temperature $T$)};\n")
f.save(__file__)
