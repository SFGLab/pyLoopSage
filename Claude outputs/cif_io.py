"""Minimal mmCIF / PSF writers (one bead = one residue, chain bonds as connections)."""
import numpy as np

_ATOM_HEAD = """data_nucsim
#
_entry.id nucsim
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.296
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
# ----------- ATOMS ----------------
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""

_CONN_HEAD = """#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
"""


def write_mmcif(points, path) -> None:
    """Write bead coordinates (angstrom units, as the CIF format expects)."""
    n = len(points)
    atoms = "".join(f"ATOM {i + 1} D CA . ALA A 1 {i + 1} ? {x:.3f} {y:.3f} {z:.3f}\n"
                    for i, (x, y, z) in enumerate(np.asarray(points)))
    bonds = "".join(f"C{i + 1} covale ALA A {i + 1} CA ALA A {i + 2} CA\n" for i in range(n - 1))
    with open(path, "w") as f:
        f.write(_ATOM_HEAD + atoms + _CONN_HEAD + bonds)


def write_psf(n: int, path, title="LoopSage chain") -> None:
    """Topology for viewing the DCD trajectory in VMD/PyMOL."""
    lines = ["PSF CMAP\n", "\n", "      1 !NTITLE\n", f"REMARKS {title}\n", "\n", f"{n:>8} !NATOM\n"]
    lines += [f"{k:>8} BEAD {k:<5} ALA  CA   A      0.000000        1.00 0           0\n" for k in range(1, n + 1)]
    lines += ["\n", f"{n - 1:>8} !NBOND: bonds\n"] + [f"{i:>8}{i + 1:>8}\n" for i in range(1, n)]
    with open(path, "w") as f:
        f.writelines(lines)
