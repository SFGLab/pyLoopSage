"""Render the collected figures/tables into one self-contained report.html."""
import base64
import datetime
import html
from pathlib import Path

from reporting.registry import REGISTRY, SECTIONS, Fig
from reporting.texts import GLOSSARY, SECTION_INTRO
from settings.schema import PARAMS, SECTIONS as PARAM_SECTIONS

esc = html.escape
HTML_DIR = Path(__file__).resolve().parents[1] / "html"   # CSS + page template


def _figure(f: Fig) -> str:
    img = base64.b64encode(f.png).decode()
    return (f'<div class="card"><h3>{esc(f.title)}</h3><figure><img alt="{esc(f.title)}" '
            f'src="data:image/png;base64,{img}"><figcaption>{esc(f.caption)}</figcaption></figure>'
            f'<details open><summary>How to read this figure</summary><p>{f.explain}</p></details></div>')


def _table_html(columns, rows) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table(t) -> str:
    cap = f'<p class="caption">{esc(t.caption)}</p>' if t.caption else ""
    return f'<div class="card"><h3>{esc(t.title)}</h3>{_table_html(t.columns, t.rows)}{cap}</div>'


def _overview() -> str:
    stats = "".join(f'<div class="stat"><small>{esc(k)}</small><span>{esc(v)}</span></div>' for k, v in REGISTRY.facts)
    timing = _table_html(["Stage", "Wall time"], [[s, t] for s, t in REGISTRY.timings])
    return (f'<section id="overview"><h2>Overview</h2><p class="intro">Headline numbers of this run, and how long each '
            f'stage took. Figures come with a short explanation; unfamiliar terms are defined in the glossary.</p>'
            f'<div class="cards">{stats}</div><div class="card"><h3>Run time per stage</h3>{timing}</div></section>')


def _parameters(cfg) -> str:
    values, tags = cfg.as_dict(), {"default": "", "ini": "ini", "cli": "cli"}
    cards = []
    for sec, title in PARAM_SECTIONS.items():
        rows = "".join(
            f"<tr><td><code>{p.name}</code></td><td>{esc(str(values[p.name] if values[p.name] is not None else ''))}</td>"
            f"<td>{esc(p.unit)}</td><td><span class='tag {tags[cfg.source[p.name]]}'>{cfg.source[p.name]}</span></td>"
            f"<td>{esc(p.help)}</td></tr>" for p in PARAMS if p.section == sec)
        cards.append(f'<details class="card"><summary>{esc(title)}</summary><table><thead><tr><th>Parameter</th>'
                     f'<th>Value</th><th>Unit</th><th>Source</th><th>Description</th></tr></thead><tbody>{rows}</tbody></table></details>')
    return ('<section id="parameters"><h2>Parameters</h2><p class="intro">Every setting used. <em>Source</em> shows whether a '
            'value is the default or came from the config file (ini) or the command line (cli).</p>' + "".join(cards) + "</section>")


def _glossary() -> str:
    items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in GLOSSARY)
    return f'<section id="glossary"><h2>Glossary</h2><div class="card"><dl>{items}</dl></div></section>'


def build_report(cfg, title="LoopSage report") -> str:
    """Assemble the full HTML document."""
    nav = [("overview", "Overview")] + [(s, t) for s, t in SECTIONS.items() if REGISTRY.of(s)] \
        + [("parameters", "Parameters"), ("glossary", "Glossary")]
    body = [_overview()]
    for sec, name in SECTIONS.items():
        items = REGISTRY.of(sec)
        if items:
            cards = "".join(_figure(i) if isinstance(i, Fig) else _table(i) for i in items)
            body.append(f'<section id="{sec}"><h2>{name}</h2><p class="intro">{SECTION_INTRO[sec]}</p>{cards}</section>')
    body += [_parameters(cfg), _glossary()]
    links = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in nav)
    chips = "".join(f'<span class="chip"><small>{esc(k)}</small>{esc(v)}</span>' for k, v in REGISTRY.facts[:4])
    page = (HTML_DIR / "template.html").read_text(encoding="utf-8")
    fill = dict(title=esc(title), css=(HTML_DIR / "style.css").read_text(encoding="utf-8"), nav=links, chips=chips,
                stamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), body="".join(body))
    for key, val in fill.items():
        page = page.replace("{{%s}}" % key, val)
    return page
