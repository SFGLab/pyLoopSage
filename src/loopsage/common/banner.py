"""Rotating one-liners shown under the banner and during progress bars.

- LoopSage banner: gradient logo, loop-extrusion drawing and a quote.
"""
import random

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from loopsage.logger import console


# ---- Rotating one-liners shown under the banner and during progress bars ----

QUOTES = [
    ("Bazinga!", "Sheldon Cooper"),
    ("That's my spot.", "Sheldon Cooper"),
    ("I'm not crazy. My mother had me tested.", "Sheldon Cooper"),
    ("Knock, knock, knock, Penny.", "Sheldon Cooper"),
    ("Fun with Flags!", "Sheldon Cooper"),
    ("Scientific progress is 90% patience and 10% Bazinga.", "Sheldon-style"),
    ("Schrodinger's cat walks into a bar. And doesn't.", "physics joke"),
    ("There are 10 kinds of people: those who understand binary and those who don't.", "science joke"),
    ("A neutron walks into a bar: 'How much for a drink?' 'For you, no charge.'", "physics joke"),
    ("Never trust atoms. They make up everything.", "chemistry joke"),
    ("Heisenberg gets pulled over. 'Do you know how fast you were going?' 'No, but I know exactly where I am.'", "physics joke"),
    ("Statistics: the art of torturing data until it confesses.", "folk wisdom"),
    ("Correlation is not causation, but it does wave suggestively.", "statistics joke"),
    ("My chromatin is a mess; my loops are fine.", "Sheldon-style"),
    ("In theory there is no difference between theory and practice. In practice there is.", "folk wisdom"),
    ("The cohesin extrudes; the physicist waits.", "Sheldon-style"),
    ("Why can't you trust chromatin? It's always up to something at the loop.", "biology joke"),
    ("I told my cohesin a joke about extrusion. It just kept going.", "biology joke"),
    ("What do you call a loop that never ends? A topologically associating bore-dom.", "biology joke"),
    ("CTCF walks into a bar. Both anchors say: 'convergent evening, sir.'", "genomics joke"),
    ("Two atoms are walking down the street. 'I lost an electron!' 'Are you sure?' 'Yes, I'm positive.'", "chemistry joke"),
    ("Helium walks into a bar. The bartender says 'We don't serve noble gases.' Helium doesn't react.", "chemistry joke"),
    ("A photon checks into a hotel. 'Any luggage?' 'No, I'm travelling light.'", "physics joke"),
    ("Why do physicists love Monte Carlo? Because they can finally gamble with a clear conscience.", "physics joke"),
    ("Monte Carlo methods: because sometimes the best way to solve it is to roll the dice a million times.", "science joke"),
    ("My simulation converged! (It converged to the wrong answer, but it converged.)", "simulation joke"),
    ("Burn-in: the polite name for 'ignore everything the sampler said at first'.", "simulation joke"),
    ("p < 0.05: the three most dangerous characters in science.", "statistics joke"),
    ("A statistician drowned in a river with an average depth of 10 cm.", "statistics joke"),
    ("Three statisticians go duck hunting. The first misses left, the second right. The third: 'We got it!'", "statistics joke"),
    ("There are only two hard things in computer science: cache invalidation, naming things, and off-by-one errors.", "programmer joke"),
    ("It works on my machine. - every simulation, ever", "programmer joke"),
    ("99 little bugs in the code. Take one down, patch it around... 127 little bugs in the code.", "programmer joke"),
    ("A SQL query walks into a bar, sees two tables and asks: 'Can I join you?'", "programmer joke"),
    ("Dark matter: the only thing physicists are certain about is that they can't find it.", "physics joke"),
    ("Why is the Navier-Stokes problem so hard? The solution keeps flowing away.", "math joke"),
    ("Old mathematicians never die; they just lose some of their functions.", "math joke"),
    ("Why did the biologist bring a ladder to the lab? To study high-level structures.", "biology joke"),
    ("I'd tell you a chemistry joke, but all the good ones Argon.", "chemistry joke"),
    ("Parallel lines have so much in common. It's a shame they'll never meet.", "math joke"),
    ("What did the polymer say to the enzyme? 'Stop pushing my buttons, I'm already a chain reaction.'", "chemistry joke"),
    ("Sheldon-style: I'm not saying I'm the best physicist here, but please check who has the whiteboard.", "Sheldon-style"),
    ("Sheldon-style: Fun with Chromatin! Episode 1: it's folded. Episode 2: still folded.", "Sheldon-style"),
    ("My knots have knots. Please do not ask Topoly.", "topology joke"),
    ("A topologist can't tell a donut from a coffee mug, so he drinks from both.", "topology joke"),
    ("Why did the DNA go to therapy? Too many unresolved loops.", "biology joke"),
    ("Entropy isn't what it used to be.", "physics joke"),
]


def pick(exclude=None):
    """Random (text, author), avoiding `exclude` when possible."""
    pool = [q for q in QUOTES if q != exclude] or QUOTES
    return random.choice(pool)


# ---- LoopSage banner: gradient logo, loop-extrusion drawing and a quote ----

_LOGO = r"""
██╗      ██████╗  ██████╗ ██████╗ ███████╗ █████╗  ██████╗ ███████╗
██║     ██╔═══██╗██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║     ██║   ██║██║   ██║██████╔╝███████╗███████║██║  ███╗█████╗
██║     ██║   ██║██║   ██║██╔═══╝ ╚════██║██╔══██║██║   ██║██╔══╝
███████╗╚██████╔╝╚██████╔╝██║     ███████║██║  ██║╚██████╔╝███████╗
╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝""".strip("\n")

_LOOPS = r"""
            ╭────────────────────────────────────╮
            │        ╭──────────────────╮        │
            │        │     ╭────────╮   │        │
 ●─●─●─●─●──●─●─●─●─●┴─●─●─●┴─●─●─●─●─●┴─●─●─●─●─●┴─●─●─●─●─●─●─●""".strip("\n")

SPLIT = 34                                  # column where "LOOP" ends and "SAGE" starts in the logo
LOOP_COLOR = "#f4f2f8"                      # "Loop": black in the report, white here (the banner panel is dark)
SAGE_GRADIENT = ["#b79cf0", "#9d7df0", "#8560e6", "#6a3de8"]   # "Sage": purple


def _paint(block: str, colors, split=None) -> Text:
    """Colour a multi-line block with a left-to-right gradient; with `split`, left part is LOOP_COLOR and the gradient covers the rest."""
    out = Text(no_wrap=True)
    lines = block.splitlines()
    width = max(len(l) for l in lines)
    for line in (l.ljust(width) for l in lines):
        lo = split or 0
        for i, ch in enumerate(line):
            if split is not None and i < split:
                out.append(ch, style=f"bold {LOOP_COLOR}")
            else:
                out.append(ch, style=colors[min((i - lo) * len(colors) // (width - lo), len(colors) - 1)])
        out.append("\n")
    out.rstrip()
    return out


def print_banner(version: str = None) -> None:
    """Show the logo, a loop sketch and a random quote."""
    text, author = pick()
    quote = Text(f"“{text}”", style="italic #c9c6d3", justify="center")
    quote.append(f"  - {author}", style="#9d7df0")
    from loopsage.version import __version__
    version = version or __version__
    sub = Text(f"stochastic loop extrusion + 3D chromatin dynamics  ·  v{version}", style="bold #b79cf0", justify="center")
    body = Group(Align.center(_paint(_LOOPS, ["#b79cf0", "#6a3de8"])), Text(""), Align.center(_paint(_LOGO, SAGE_GRADIENT, SPLIT)),
                 Text(""), sub, Text(""), quote)
    console.print(Panel(body, border_style="#6a3de8", style="on #0d0b14", padding=(1, 3), expand=False))
