# Demos

Tiny synthetic runs (`data/demo_loops.bedpe`: 60 loops on chr1:77-95 Mb, made by `make_demo_data.py`).

| Command | What it shows | Time |
|---|---|---|
| `python demo/run_demo.py quick` | input, loop extrusion (MC), validation | seconds |
| `python demo/run_demo.py full` | + 3D polymer MD, geometry, 3D viewer and videos, map features, linking analysis | about 15 seconds |
| `python demo/run_demo.py epigenetic` | loop extrusion + the optional epigenetic model (dynamic coupling), with its extra figures | about 20 seconds |
| `python demo/run_demo.py speed` | LEFs with Maxwell-Boltzmann speeds: velocity statistics, phase space and free-energy flow figures | about 15 seconds |
| `loopsage -c demo/demo_quick.ini -vv` | same as `quick`, with info messages | seconds |

Each run writes a folder `results/chr1_77000000_95000000_<n>/` containing `report/report.html`.
Add `-v` (tables) or `-vv` (info) to `run_demo.py` to see more in the terminal.
