# LoopSage documentation

Notes for **LoopSage**, a program that turns chromatin loop data into ensembles of 3D genome structures.
Start with the [home page](intro/index.md), or pick a chapter below.

| Chapter | Pages | Status |
|---|---|---|
| **Intro** | [Home: what LoopSage is about](intro/index.md) · [Installation](intro/installation.md) | written |
| **Theory** | [Theory](theory/index.md) | coming soon |
| **Implementation** | [Implementation](implementation/index.md) | coming soon |
| **Reference** | [Overview](reference/index.md) · [Command-line interface](reference/cli.md) · [Config files](reference/config.md) | written |
| **Acknowledgments** | [Acknowledgments](acknowledgments/index.md) | coming soon |

## Read as a website

The same files build into a searchable site with [MkDocs](https://www.mkdocs.org) (configuration: `mkdocs.yml` in the repository root).

```bash
pip install mkdocs-material
mkdocs serve        # open http://127.0.0.1:8000
```

## Keeping the reference up to date

`reference/cli.md` and `reference/config.md` are generated from the parameter list in `src/settings/schema.py`. After changing a parameter, regenerate them:

```bash
PYTHONPATH=src python tools/make_reference.py
```
