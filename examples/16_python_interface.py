"""Python interface: the same workflow as 14_motif_optimizer_compartments.ini, one function per step.
Run: python examples/16_python_interface.py   (edit the file names first)."""
import loopsage as ls

cfg = ls.config(region="chr1:75000000-95000000", n_beads=2000, seed=1,
                motif_genome="hg38.fa", epi_coeff=1.0, do_compartments=True)   # all settings in one Config
geom = ls.geometry(cfg)                                                       # bead grid
inputs = ls.read_inputs(cfg, geom, loops="loops.bedpe", compartments="compartments.bed")

opt = ls.optimize(cfg, inputs, geom)                  # Nelder-Mead on short MC runs
cfg = ls.apply_optimum(cfg, opt)                      # config with the fitted values

mc = ls.run_mc(cfg, inputs, geom)                     # loop extrusion
md = ls.run_md(cfg, mc, geom, folder="results_py")    # 3D structures (.cif frames) saved here

ls.plots.energy_traces(mc)                            # matplotlib figure
