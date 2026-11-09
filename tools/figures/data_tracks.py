"""Writes data_tracks.tex: the input files on one toy region, and what each feeds in the model (toy data).

Run: python3 data_tracks.py
"""
from pathlib import Path

import numpy as np

N, W0, WW = 40, 3.1, 10.2
X = lambda i: W0 + WW * (i + 0.5) / N
rng = np.random.default_rng(3)
SITES = {4: "f", 14: "f", 25: "r", 35: "r"}                       # CTCF anchors and their orientation (toy)
LOOPS = [(4, 25, 3), (14, 25, 2), (4, 14, 1), (25, 35, 2)]        # left bead, right bead, weight
bump = lambda c, w: np.exp(-0.5 * ((np.arange(N) - c) / w) ** 2)
bw = sum(bump(c, 1.1) * a for c, a in ((4, 1), (14, 0.7), (25, 1), (35, 0.6))) + 0.08 * rng.random(N)
lef = bump(10, 4) * 0.8 + bump(30, 3) * 0.6 + 0.05 * rng.random(N)
comp = ["A"] * 12 + ["B"] * 12 + ["A"] * 8 + ["B"] * 8
i, j = np.indices((N, N))
hic = 0.5 / (1 + np.abs(i - j)) ** 0.9                              # decay with distance
for a, b, w in LOOPS:
    hic[a:b + 1, a:b + 1] += 0.06 * w                               # domain between the anchors
    hic += 0.9 * w / 3 * np.exp(-((i - a) ** 2 + (j - b) ** 2) / 1.2)   # loop dot at the corner
    hic += 0.05 * w * (np.exp(-(i - a) ** 2 / 0.5) * (j > a) * (j <= b) + np.exp(-(j - b) ** 2 / 0.5) * (i >= a) * (i < b))   # stripes
hic = np.triu(hic) + np.triu(hic, 1).T
hic = np.clip(hic / np.percentile(hic[np.triu_indices(N, 3)], 99.5), 0, 1)

t = r"""\input{preamble.tex}
\definecolor{A}{HTML}{D1303F}\definecolor{B}{HTML}{5B4FD6}
\tikzset{ln/.style={line width=0.7pt,ink!45},arc/.style={line width=1.1pt,cohesin!85},guide/.style={ink!18,dashed,line width=0.5pt}}
\begin{document}
\begin{tikzpicture}
  \panelbg{0}{-11.6}{19.6}{2.6}
"""
for b in SITES:                                                    # guides through the anchors
    t += f"  \\draw[guide] ({X(b):.2f},1.9)--({X(b):.2f},-9.2);\n"


def label(y, name, tag, req):
    """Left label: file kind, and a required / optional mark."""
    col = "cohesin" if req else "ink!55"
    return (f"  \\node[lab,anchor=east,align=right] at (2.8,{y:.2f}) {{\\textbf{{{name}}}\\\\[-1pt]\\footnotesize\\color{{ink!70}}{tag}}};"
            f"\\node[font=\\scriptsize\\bfseries,text={col},anchor=east] at (2.8,{y - 0.72:.2f}) {{{'required' if req else 'optional'}}};\n")


def use(y, head, sub):
    """Right note: what the file feeds."""
    return f"  \\node[lab,anchor=west,text width=5.3cm,align=left] at (13.9,{y:.2f}) {{\\textbf{{{head}}}\\\\[-1pt]\\footnotesize\\color{{ink!70}}{sub}}};\n"


# 1 loops
y = 0.0
t += label(0.75, "Loops", "BEDPE", True) + use(0.75, r"anchor profiles $L$, $R$, loop map $J$", "Monte Carlo: binding energy")
t += f"  \\draw[ln] ({X(0):.2f},{y})--({X(N - 1):.2f},{y});\n"
for a, b, w in LOOPS:
    h = 0.07 * (b - a) + 0.35
    t += f"  \\draw[arc,line width={0.6 + 0.5 * w}pt] ({X(a):.2f},{y}) .. controls ({X(a):.2f},{y + h:.2f}) and ({X(b):.2f},{y + h:.2f}) .. ({X(b):.2f},{y});\n"
# 2 CTCF sites with orientation
y = -1.7
t += label(y + 0.2, "CTCF sites", "BED / narrowPeak, or motif search", False).replace("anchor=east,align=right", "anchor=east,align=right,text width=2.6cm")
t += use(y + 0.2, "orientation splits $L$ and $R$", "BEDPE columns 8, 9, or a FASTA + motif")
t += f"  \\draw[ln] ({X(0):.2f},{y})--({X(N - 1):.2f},{y});\n"
for b, o in SITES.items():
    col, d = ("fwd", 1) if o == "f" else ("rev", -1)
    t += f"  \\draw[{col},line width=1.3pt] ({X(b):.2f},{y})--({X(b):.2f},{y + 0.45});\\draw[{col},line width=1.6pt,-{{Stealth[length=4.5pt,width=5pt]}}] ({X(b) - 0.22 * d:.2f},{y + 0.45})--({X(b) + 0.26 * d:.2f},{y + 0.45});\n"
# 3 BigWig, 4 loading
for y, name, tag, col, v, hd, sb in ((-3.5, "Binding tracks", "BigWig", "cohesin", bw, "extra binding term", "Monte Carlo: other factors at anchors"),
                                     (-5.0, "Loading track", "BigWig", "condensin", lef, "where rings load", "Monte Carlo: rebinding move")):
    t += label(y + 0.3, name, tag, False) + use(y + 0.3, hd, sb)
    t += f"  \\draw[ln] ({X(0):.2f},{y})--({X(N - 1):.2f},{y});\n"
    pts = f"({X(0):.2f},{y}) " + " ".join(f"({X(k):.2f},{y + 0.7 * v[k] / v.max():.2f})" for k in range(N)) + f" ({X(N - 1):.2f},{y})"
    t += f"  \\fill[{col}!35] plot[smooth,tension=0.4] coordinates {{{pts}}};\\draw[{col},line width=1pt] plot[smooth,tension=0.4] coordinates {{{pts}}};\n"
# 5 compartments
y = -6.35
t += label(y + 0.1, "Compartments", "BigWig or BED", False) + use(y + 0.1, r"field $h$: A $=+1$, B $=-1$", "Monte Carlo epigenetic term, 3D attraction")
for k, c in enumerate(comp):
    t += f"  \\fill[{c}!70] ({X(k) - WW / N / 2:.3f},{y - 0.17}) rectangle ({X(k) + WW / N / 2:.3f},{y + 0.17});\n"
t += f"  \\node[font=\\scriptsize\\bfseries,white] at ({X(5.5):.2f},{y}) {{A}};\\node[font=\\scriptsize\\bfseries,white] at ({X(17.5):.2f},{y}) {{B}};\\node[font=\\scriptsize\\bfseries,white] at ({X(27.5):.2f},{y}) {{A}};\\node[font=\\scriptsize\\bfseries,white] at ({X(35.5):.2f},{y}) {{B}};\n"
# 6 Hi-C triangle
y = -10.5
t += label(y + 1.3, "Hi-C map", ".hic", False) + use(y + 1.3, "reference only", "validation, optimizer target")
c = WW / N
for a in range(N):
    for b in range(a + 1, min(N, a + 25)):
        v = hic[a, b]
        if v > 0.03:
            cx, cy = (X(a) + X(b)) / 2, y + (b - a) * c / 2
            t += f"  \\fill[cohesin!{int(100 * min(v, 1) ** 0.55)}!white] ({cx - c / 2 - 0.01:.3f},{cy:.3f})--({cx:.3f},{cy + c / 2:.3f})--({cx + c / 2:.3f},{cy:.3f})--({cx:.3f},{cy - c / 2:.3f})--cycle;\n"
t += f"  \\draw[ln] ({X(0) - c / 2:.2f},{y})--({X(N - 1) + c / 2:.2f},{y});\n"
t += f"  \\node[note] at ({W0 + WW / 2:.2f},-11.25) {{the same region, cut into beads: every file is read onto this grid}};\n"
t += "\\end{tikzpicture}\n\\end{document}\n"
Path(__file__).with_name("data_tracks.tex").write_text(t)
