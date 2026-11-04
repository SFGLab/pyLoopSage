"""OpenMM force builders for the bead-spring chromatin model (units: nm, kJ/mol)."""
import numpy as np
import openmm as mm

BOND_LENGTH, BOND_K = 0.1, 3e5     # nm, kJ/mol/nm^2: chain connectivity
EV_EPSILON, EV_R_SMALL = 0.05, 0.1  # excluded-volume softening

# Compartment interaction between states (0 = B strong, 1 = neutral, 2 = A weak)
_COMP_TABLE = [-1.0, 0.0, 0.0,
               0.0, -0.8, 0.0,
               0.0, 0.0, -0.2]


def add_excluded_volume(system, n, strength, power):
    """Soft repulsion between all bead pairs. Returns the force (update with set_ev)."""
    f = mm.CustomNonbondedForce(f"(epsilon1*epsilon2*(sigma1*sigma2)/(r+r_small))^{power}")
    f.addGlobalParameter("r_small", EV_R_SMALL)
    f.addPerParticleParameter("sigma")
    f.addPerParticleParameter("epsilon")
    for _ in range(n):
        f.addParticle([np.sqrt(strength), EV_EPSILON])
    system.addForce(f)
    return f


def ev_profile(n, strength, p_ev, continuous):
    """Per-bead sigma; topoisomerase switches excluded volume off (0) on some beads."""
    sigma = np.full(n, np.sqrt(strength))
    if p_ev <= 0:
        return sigma
    if continuous:                                   # one stretch of p_ev * n beads
        length = max(1, int(p_ev * n))
        start = np.random.randint(0, max(n - length, 1))
        sigma[start:start + length] = 0.0
    else:                                            # each bead independently
        sigma[np.random.rand(n) < p_ev] = 0.0
    return sigma


def set_ev(force, sigma):
    for i, s in enumerate(sigma):
        force.setParticleParameters(i, [s, EV_EPSILON])


def add_bonds(system, n):
    f = mm.HarmonicBondForce()
    for i in range(n - 1):
        f.addBond(i, i + 1, BOND_LENGTH, BOND_K)
    system.addForce(f)


def add_angles(system, n, strength):
    """Stiffness: penalise bending away from a straight chain (angle pi)."""
    f = mm.HarmonicAngleForce()
    for i in range(n - 2):
        f.addAngle(i, i + 1, i + 2, np.pi, strength)
    system.addForce(f)


def add_compartments(system, states, width):
    """Block-copolymer force: same-compartment beads attract within `width` nm."""
    f = mm.CustomNonbondedForce("E(s1,s2)*exp(-(r-r0)^2/(2*sigma^2))")
    f.setForceGroup(1)
    f.addGlobalParameter("sigma", width)
    f.addGlobalParameter("r0", 0.0)
    f.addPerParticleParameter("s")
    f.addTabulatedFunction("E", mm.Discrete2DFunction(3, 3, _COMP_TABLE))
    for s in states:
        f.addParticle([int(s)])
    system.addForce(f)
    return f


def make_loop_force(ms, ns, length, k):
    """Springs pulling each loop's two anchors together."""
    f = mm.HarmonicBondForce()
    for m, n in zip(ms, ns):
        f.addBond(int(m), int(n), length, k)
    return f
