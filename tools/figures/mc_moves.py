"""Writes mc_moves.tex: the three Monte Carlo moves (rebind, slide, epigenetic state change) on a toy chain of 20 beads.

Run: python3 mc_moves.py
"""
from pathlib import Path

N, DX = 20, 0.27
head = r"""\input{preamble.tex}
\definecolor{B}{HTML}{5B4FD6}
\tikzset{ln/.style={line width=0.7pt,ink!45},arc/.style={line width=1.2pt,cohesin},ghost/.style={line width=1.0pt,ink!30,dashed},
  mv/.style={line width=1.2pt,ink!70,-{Stealth[length=5pt,width=6pt]}}}
\begin{document}
\begin{tikzpicture}
  \panelbg{0}{-0.2}{13.4}{4.4}\panelbg{0}{-4.8}{13.4}{-0.4}\panelbg{0}{-9.8}{13.4}{-5.0}
"""


def x(i, x0): return x0 + DX * i


def chain(x0, y, cols=None, r=0.085):
    """Bead line; cols = {bead: colour}."""
    s = f"  \\draw[ln] ({x(0, x0):.2f},{y})--({x(N - 1, x0):.2f},{y});\n"
    for i in range(N):
        c = (cols or {}).get(i, "ink!25")
        s += f"  \\fill[{c}] ({x(i, x0):.2f},{y}) circle ({r});\n"
    return s


def arc(m, n, x0, y, style="arc"):
    h = 0.075 * (n - m) * 4 / 3
    return f"  \\draw[{style}] ({x(m, x0):.2f},{y}) .. controls ({x(m, x0):.2f},{y + h:.2f}) and ({x(n, x0):.2f},{y + h:.2f}) .. ({x(n, x0):.2f},{y});\n"


def title(n, t, y, sub=None):
    s = f"  \\badge{{0.45}}{{{y}}}{{{n}}}\\node[hd] at (0.85,{y}) {{{t}}};\n"
    return s + (f"  \\node[note,anchor=north west,text width=12.2cm] at (0.85,{y - 0.3}) {{{sub}}};\n" if sub else "")


B0, A0 = 0.9, 7.2                                           # x of the "before" and "after" chains
tex = head
# ---- a: rebind
y = 0.9
tex += title("a", "Rebinding: a LEF falls off and loads again as a minimal loop $(m, m+2)$", 3.95, "pick a LEF, unbind it and load it at a random bead (or where the loading track is high)")
tex += chain(B0, y) + arc(5, 13, B0, y) + arc(8, 11, B0, y, "arc,cohesin!35")
tex += chain(A0, y) + arc(5, 13, A0, y, "ghost") + arc(8, 11, A0, y, "arc,cohesin!35") + arc(15, 17, A0, y)
tex += f"  \\draw[mv] ({x(13, B0) + 0.6:.2f},1.0)--({A0 - 0.6:.2f},1.0);\n  \\node[note] at (6.65,1.35) {{$\\Delta E$}};\n"
tex += f"  \\node[note] at ({x(10, B0):.2f},0.35) {{before}};\\node[note] at ({x(10, A0):.2f},0.35) {{after}};\n"
# ---- b: slide
y = -2.95
tex += title("b", "Sliding: each leg moves by one bead (a random step, left or right)", -0.95, "the loop grows or shrinks; a leg that would land on another loop's leg stays put, and a loop never gets shorter than 2 beads")
tex += chain(B0, y) + arc(6, 13, B0, y)
tex += chain(A0, y) + arc(6, 13, A0, y, "ghost") + arc(5, 14, A0, y)
for i, d in ((6, -1), (13, 1)):
    tex += f"  \\draw[mv] ({x(i, B0):.2f},{y - 0.3})--({x(i, B0) + 0.35 * d:.2f},{y - 0.3});\n"
tex += f"  \\draw[mv] ({x(13, B0) + 0.6:.2f},{y + 0.1})--({A0 - 0.6:.2f},{y + 0.1});\n"
tex += f"  \\node[note] at ({x(10, B0):.2f},{y - 0.7}) {{before}};\\node[note] at ({x(10, A0):.2f},{y - 0.7}) {{after (both legs moved out)}};\n"
# ---- c: epigenetic change
y = -7.5
tex += title("c", "Epigenetic move: a bead takes another state", -5.55, "chosen instead of a LEF move with probability \\texttt{p\\_epigenetic}; linked beads prefer equal states")
cols = {i: "B" for i in range(N)} | {i: "fwd" for i in (4, 5, 6, 12, 14)}
cols2 = cols | {13: "fwd"}
tex += chain(B0, y, cols, 0.1) + arc(5, 13, B0, y, "arc,cohesin!60")
tex += chain(A0, y, cols2, 0.1) + arc(5, 13, A0, y, "arc,cohesin!60")
tex += f"  \\draw[mv] ({x(13, B0) + 0.6:.2f},{y + 0.1})--({A0 - 0.6:.2f},{y + 0.1});\n"
tex += f"  \\node[note] at ({x(10, B0):.2f},{y - 0.6}) {{before: bead 13 is B, its partners are A}};\\node[note] at ({x(10, A0):.2f},{y - 0.6}) {{after: it agrees with them, so $\\Delta E<0$}};\n"
tex += f"  \\node[note,anchor=west] at (0.85,-9.3) {{states: \\textcolor{{fwd}}{{$\\bullet$}} A (red) \\quad \\textcolor{{B}}{{$\\bullet$}} B (blue)}};\n  \\node[note,anchor=east] at (12.9,-9.3) {{every move is accepted with probability $\\min(1,e^{{-\\Delta E/T}})$}};\n"
tex += "\\end{tikzpicture}\n\\end{document}\n"
Path(__file__).with_suffix(".tex").write_text(tex)
