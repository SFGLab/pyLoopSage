"""Config: precedence defaults < ini < CLI, old flat configs, validation."""
import pytest

from loopsage.settings.loader import load_config
from loopsage.settings.schema import BY_NAME, PARAMS


def test_every_cli_flag_maps_to_a_param():
    assert all(p.flag.startswith("--") for p in PARAMS if p.flag)
    assert len({p.flag for p in PARAMS if p.flag}) == len([p for p in PARAMS if p.flag])


def test_cli_only(tiny_args):
    cfg = load_config(tiny_args)
    assert cfg.chrom == "chr1" and cfg.region_start == 77_000_000 and cfg.n_beads == 120
    assert cfg.run_md is False and cfg.source["n_steps"] == "cli"


def test_ini_overrides_default_and_cli_overrides_ini(demo_bedpe, tmp_path):
    ini = tmp_path / "c.ini"
    ini.write_text(f"[input]\ninteraction_file = {demo_bedpe}\nchrom = chr1\n[monte_carlo]\nn_steps = 5000\nburnin = 100\nmc_step = 100\n")
    cfg = load_config(["-c", str(ini)])
    assert cfg.n_steps == 5000 and cfg.source["n_steps"] == "ini"
    cfg = load_config(["-c", str(ini), "--mc-steps", "8000"])
    assert cfg.n_steps == 8000 and cfg.source["n_steps"] == "cli"


def test_old_flat_main_section_and_deprecated_key(demo_bedpe, tmp_path):
    ini = tmp_path / "old.ini"
    ini.write_text(f"[Main]\nINTERACTION_FILE = {demo_bedpe}\nCHROM = chr1\nN_STEPS = 3000\nBURNIN = 100\nMC_STEP = 100\nSAVE_PLOTS = True\n")
    cfg = load_config(["-c", str(ini)])
    assert cfg.n_steps == 3000
    assert any("save_plots" in n for n in cfg.notes)


def test_set_escape_hatch(tiny_args):
    cfg = load_config(tiny_args + ["--set", "ev_p=0.1"])
    assert cfg.ev_p == pytest.approx(0.1)


@pytest.mark.parametrize("bad", [["--burnin", "999999"], ["--sample-every", "0"]])
def test_invalid_values_exit(tiny_args, bad):
    with pytest.raises(SystemExit):
        load_config(tiny_args + bad)


def test_missing_input_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_config(["-f", str(tmp_path / "nope.bedpe"), "-r", "chr1:1-100000"])


def test_defaults_documented():
    assert all(p.help for p in PARAMS) and BY_NAME["run_md"].flag == "--run-MD"


def test_binding_mode_replaces_contrastive_binding(tmp_path):
    bedpe = tmp_path / "loops.bedpe"
    bedpe.write_text("chr1\t1000\t1100\tchr1\t1500\t1600\t5\n")
    base = ["-f", str(bedpe), "-r", "chr1:1000-2000"]
    assert load_config(base).binding_mode == "preferential"
    assert load_config(base + ["--binding-mode", "contrastive"]).binding_mode == "contrastive"
    old = load_config(base + ["--set", "contrastive_binding=true"])                 # old key still works and says so
    assert old.binding_mode == "contrastive" and any("binding_mode" in n for n in old.notes)
    ini = tmp_path / "old.ini"
    ini.write_text("[energy]\ncontrastive_binding = False\n")
    assert load_config(base + ["-c", str(ini)]).binding_mode == "preferential"
