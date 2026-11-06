"""Small helpers that write TikZ code for the theory figures (panels, axes, curves, bars)."""
from pathlib import Path

import numpy as np

HEAD = r"""\input{preamble.tex}
\definecolor{A}{HTML}{D1303F}\definecolor{B}{HTML}{5B4FD6}
\tikzset{ln/.style={line width=0.7pt,ink!45},cur/.style={line width=1.8pt,cohesin},thin/.style={line width=0.9pt},
  mv/.style={line width=1.2pt,ink!70,-{Stealth[length=5pt,width=6pt]}},box/.style={rounded corners=5pt,draw=cohesin!70,fill=white,line width=0.9pt,align=center,font=\small,text=ink,inner sep=4pt}}
\begin{document}
\begin{tikzpicture}
"""
TAIL = "\\end{tikzpicture}\n\\end{document}\n"


class Fig:
    """Collects TikZ lines. Panels: panel(letter, title, x0, y0, x1, y1) with the title at the top left."""

    def __init__(self):
        self.s = []

    def add(self, t):
        self.s.append(t)

    def panel(self, letter, title, x0, y0, x1, y1):
        self.add(f"  \\panelbg{{{x0}}}{{{y0}}}{{{x1}}}{{{y1}}}\\badge{{{x0 + 0.45}}}{{{y1 - 0.45}}}{{{letter}}}\\node[hd] at ({x0 + 0.85},{y1 - 0.45}) {{{title}}};\n")

    def text(self, x, y, t, style="note", extra=""):
        self.add(f"  \\node[{style}{extra}] at ({x:.2f},{y:.2f}) {{{t}}};\n")

    def save(self, path):
        Path(path).with_suffix(".tex").write_text(HEAD + "".join(self.s) + TAIL)


class Axes:
    """Data -> cm mapping with an L-shaped frame, ticks and labels."""

    def __init__(self, fig, x0, y0, w, h, xlim, ylim, xlabel="", ylabel="", xticks=(), yticks=(), xfmt="{:g}", yfmt="{:g}", log=False):
        self.f, self.x0, self.y0, self.w, self.h, self.xlim, self.ylim, self.log = fig, x0, y0, w, h, xlim, ylim, log
        fig.add(f"  \\draw[ln] ({x0},{y0})--({x0 + w},{y0});\\draw[ln] ({x0},{y0})--({x0},{y0 + h});\n")
        for t in xticks:
            X, _ = self.p(t, ylim[0])
            fig.add(f"  \\draw[ln] ({X:.2f},{y0})--({X:.2f},{y0 - 0.08});\\node[note] at ({X:.2f},{y0 - 0.3}) {{{xfmt.format(t)}}};\n")
        for t in yticks:
            _, Y = self.p(xlim[0], t)
            fig.add(f"  \\draw[ln] ({x0},{Y:.2f})--({x0 - 0.08},{Y:.2f});\\node[note,anchor=east] at ({x0 - 0.1},{Y:.2f}) {{{yfmt.format(t)}}};\n")
        if xlabel:
            fig.add(f"  \\node[note] at ({x0 + w / 2:.2f},{y0 - 0.75}) {{{xlabel}}};\n")
        if ylabel:
            fig.add(f"  \\node[note,rotate=90] at ({x0 - 0.85:.2f},{y0 + h / 2:.2f}) {{{ylabel}}};\n")

    def p(self, x, y):
        (a, b), (c, d) = self.xlim, self.ylim
        return self.x0 + self.w * (x - a) / (b - a), self.y0 + self.h * (y - c) / (d - c)

    def line(self, xs, ys, style="cur", every=1):
        pts = " ".join("({:.3f},{:.3f})".format(*self.p(x, y)) for x, y in list(zip(xs, ys))[::every])
        self.f.add(f"  \\draw[{style}] plot[smooth,tension=0.5] coordinates {{{pts}}};\n" if style.startswith("sm") else f"  \\draw[{style}] plot coordinates {{{pts}}};\n")

    def poly(self, xs, ys, style):
        pts = " -- ".join("({:.3f},{:.3f})".format(*self.p(x, y)) for x, y in zip(xs, ys))
        self.f.add(f"  \\draw[{style}] {pts};\n")

    def fill(self, xs, ys, style):
        pts = " -- ".join("({:.3f},{:.3f})".format(*self.p(x, y)) for x, y in zip(xs, ys))
        self.f.add(f"  \\fill[{style}] {pts} -- cycle;\n")

    def bars(self, xs, ys, wd, style):
        for x, y in zip(xs, ys):
            X0, Y0 = self.p(x - wd / 2, self.ylim[0]); X1, Y1 = self.p(x + wd / 2, y)
            self.f.add(f"  \\fill[{style}] ({X0:.3f},{Y0:.3f}) rectangle ({X1:.3f},{Y1:.3f});\n")

    def hline(self, y, style="ln,dashed"):
        (X0, Y), (X1, _) = self.p(self.xlim[0], y), self.p(self.xlim[1], y)
        self.f.add(f"  \\draw[{style}] ({X0:.2f},{Y:.2f})--({X1:.2f},{Y:.2f});\n")

    def vline(self, x, style="ln,dashed"):
        (X, Y0), (_, Y1) = self.p(x, self.ylim[0]), self.p(x, self.ylim[1])
        self.f.add(f"  \\draw[{style}] ({X:.2f},{Y0:.2f})--({X:.2f},{Y1:.2f});\n")

    def label(self, x, y, t, style="note", extra=""):
        X, Y = self.p(x, y)
        self.f.text(X, Y, t, style, extra)
