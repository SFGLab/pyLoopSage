"""Config contract: every command-line option is also a config.ini key, and config.ini offers more (advanced) keys."""
import pytest

from loopsage.settings.loader import build_parser, load_config, read_ini
from loopsage.settings.schema import BY_NAME, PARAMS

# CLI options that are not parameters themselves: help, the config file, repeatable --set, region (= 3 ini keys), -v (= verbosity)
SPECIAL = {"help", "config", "set", "region", "verbose"}


@pytest.fixture(scope="module")
def files(tmp_path_factory, demo_bedpe):
    d = tmp_path_factory.mktemp("cfgfiles")
    fa, pfm = d / "g.fa", d / "m.pfm"
    fa.write_text(">chr1\nACGT\n")
    pfm.write_text("1 2 3\n1 2 3\n1 2 3\n1 2 3\n")
    return dict(interaction_file=str(demo_bedpe), motif_genome=str(fa), motif_pfm=str(pfm), out_path=str(d / "out"))


def _cli_dests():
    return {a.dest for a in build_parser()._actions} - SPECIAL


def _sample(p, files):
    """A valid non-default value (as text) for one parameter."""
    if p.name in files:
        return files[p.name]
    if p.choices:
        return next(c for c in p.choices if c != p.default)
    if p.kind == "bool":
        return str(not p.default)
    if p.kind == "int":
        return str((p.default or 2) + 1) if p.name not in ("n_beads", "run_number", "seed") else "7"
    if p.kind == "float":
        return str(round((p.default or 1.0) * 0.5, 6))
    return "x"


def test_every_cli_option_is_a_config_key():
    dests = _cli_dests()
    assert dests and dests <= set(BY_NAME)                                # flags map onto schema parameters
    assert {"chrom", "region_start", "region_end", "verbosity"} <= set(BY_NAME)    # -r and -v are covered too
    ini_keys = set(read_ini("config_example.ini"))
    assert dests <= ini_keys and {"chrom", "region_start", "region_end", "verbosity"} <= ini_keys


def test_config_has_more_parameters_than_the_cli():
    advanced = [p.name for p in PARAMS if not p.flag]
    assert len(PARAMS) > len(_cli_dests()) and len(advanced) >= 10
    assert {"motif_pseudocount", "motif_pad", "lef_drift"} <= set(advanced) or {"motif_pseudocount", "motif_pad"} <= set(advanced)


def test_example_config_lists_every_parameter():
    assert set(read_ini("config_example.ini")) == {p.name for p in PARAMS}


@pytest.mark.parametrize("name", sorted(_cli_dests()))
def test_cli_and_ini_give_the_same_value(name, files, tmp_path, demo_bedpe):
    p = BY_NAME[name]
    val = _sample(p, files)
    base = {"interaction_file": files["interaction_file"], "chrom": "chr1", "region_start": "77000000", "region_end": "95000000"}
    via_cli = load_config(["-f", files["interaction_file"], "-r", "chr1:77000000-95000000"] + ([] if name in base else
                                                                                         [p.flag, val] if p.kind != "bool" else [f"--{'' if val == 'True' else 'no-'}{p.flag[2:]}"]))
    ini = tmp_path / "c.ini"
    ini.write_text("[input]\n" + "".join(f"{k} = {v}\n" for k, v in {**base, **({} if name in base else {name: val})}.items()))
    via_ini = load_config(["-c", str(ini)])
    assert getattr(via_cli, name) == getattr(via_ini, name)
    if name not in base:
        assert via_ini.source[name] == "ini" and via_cli.source[name] == "cli"


def test_advanced_ini_only_keys_work_and_cli_overrides_ini(files, tmp_path):
    ini = tmp_path / "c.ini"
    ini.write_text(f"[input]\ninteraction_file = {files['interaction_file']}\nchrom = chr1\nregion_start = 77000000\nregion_end = 95000000\n"
                   "motif_pseudocount = 0.5\nmotif_pad = 200\nmin_pet = 2\n[output]\nverbosity = 2\n")
    cfg = load_config(["-c", str(ini)])
    assert cfg.motif_pseudocount == 0.5 and cfg.motif_pad == 200 and cfg.min_pet == 2 and cfg.verbosity == 2
    over = load_config(["-c", str(ini), "--min-pet", "5", "-v"])
    assert over.min_pet == 5 and over.verbosity == 1 and over.motif_pad == 200      # -v beats the ini value; other keys kept
    assert load_config(["-c", str(ini), "-s", "motif_pad=50"]).motif_pad == 50


def test_resolved_config_is_a_valid_config(files, tmp_path):
    """The ini written at the start of a run can be loaded again and gives the same values."""
    from loopsage.settings.loader import write_ini
    cfg = load_config(["-f", files["interaction_file"], "-r", "chr1:77000000-95000000", "--min-pet", "3", "--lef-speed", "maxwell"])
    out = tmp_path / "resolved.ini"
    write_ini(out, cfg.as_dict(), cfg.source)
    again = load_config(["-c", str(out)])
    assert all(getattr(cfg, p.name) == getattr(again, p.name) for p in PARAMS)
