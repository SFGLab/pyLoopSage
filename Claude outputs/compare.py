"""Validate a simulated contact map against the experimental input file."""
from dataclasses import dataclass, field

from helpers.matrices import insulation_score, ps_curve
from helpers.stats import normalize_signal
from logger import get_logger
from preproc.readers import read_region_table
from validation import orientation as ori
from validation import signals as sig
from validation.stats import correlate

log = get_logger(__name__)


@dataclass
class ValidationResult:
    rows: list = field(default_factory=list)         # one dict per correlation (see correlate)
    plots: list = field(default_factory=list)        # (key, signal name, xlabel, correlation dict)
    diagnostics: list = field(default_factory=list)  # orientation tests
    notes: list = field(default_factory=list)


def _add(res, label, signal, key, c, xlabel, plot=True):
    if c is None:
        res.notes.append(f"{label} ({signal}): too little data.")
        return
    res.rows.append(dict(label=label, signal=signal, **{k: v for k, v in c.items() if k not in ("exp", "sim")}))
    if plot:
        res.plots.append((key, signal, xlabel, c))


def validate(mat, cfg, geom) -> ValidationResult:
    """Correlate simulated vs experimental loop strength, P(s) decay and insulation."""
    fmt = sig.detect_format(cfg.interaction_file)
    df = read_region_table(cfg.interaction_file, geom, paired=(fmt == "bedpe"))
    res, bead_x = ValidationResult(), "Genomic position (bead index)"
    paired = fmt == "bedpe"

    if paired:
        x, y, e_oe, s_oe = sig.oe_values(df, mat, geom)
        exp, sim = sig.aggregate_to_beads(x, y, e_oe, s_oe, geom.n_beads)
        name = "loop strength (O/E)"
    else:
        exp, sim = sig.site_signal(mat, df, geom)
        name = "site strength"
    en, sn = normalize_signal(exp), normalize_signal(sim)
    _add(res, "Primary (zero-excluded)", name, "pearson", correlate(en, sn, exp, sim, True, 500), bead_x)
    _add(res, "Reference (zeros kept)", name, "", correlate(en, sn, drop_zero=False), bead_x, plot=False)

    if paired and len(x) > 1:
        _add(res, "Per-loop (no aggregation)", "loop strength (O/E)", "pearson_direct",
             correlate(normalize_signal(e_oe), normalize_signal(s_oe), e_oe, s_oe, True, 500), "Loop index")
        ps_exp, ps_sim = sig.exp_ps(df, geom), ps_curve(mat)
        _add(res, "P(s) decay", "contact vs bead separation", "ps_decay",
             correlate(normalize_signal(ps_exp), normalize_signal(ps_sim), ps_exp, ps_sim), "Separation s (beads)")
        ins = insulation_score(mat)
        _add(res, "Boundary proxy", "anchor density vs insulation", "insulation",
             correlate(en, normalize_signal(ins), exp, ins), bead_x)
        res.diagnostics.append(ori.canonical_consistency(mat, df, geom))
    res.diagnostics.append(ori.asymmetry_consistency(mat, df, fmt, geom))
    res.diagnostics = [d for d in res.diagnostics if d]
    log.info("Validation: %d correlations, %d diagnostics", len(res.rows), len(res.diagnostics))
    return res
