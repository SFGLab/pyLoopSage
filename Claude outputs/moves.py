"""Monte Carlo proposals for one LEF: rebinding and sliding."""
import numpy as np
from numba import njit


@njit(cache=True)
def rebind(p):
    """Unbind and load a new minimal loop (m, m+2), optionally at preferred loading sites."""
    if p.use_track:
        m = min(np.searchsorted(p.cdf, np.random.rand()), p.N_beads - 3)
    else:
        m = np.random.randint(0, p.N_beads - 3)
    return m, m + 2


@njit(cache=True)
def slide(p, m_old, n_old, ms, ns):
    """Move each leg by +-1..step_length beads (outward only if rw is False)."""
    dm, dn = 1, 1
    if p.step_length > 1:
        dm, dn = 1 + np.random.randint(0, p.step_length), 1 + np.random.randint(0, p.step_length)
    s1, s2 = -1, 1
    if p.rw:
        s1 = -1 if np.random.rand() < 0.5 else 1
        s2 = -1 if np.random.rand() < 0.5 else 1
    m_new = max(0, m_old + s1 * dm)
    n_new = min(p.N_beads - 1, n_old + s2 * dn)
    for k in range(ms.shape[0]):          # a leg may not land on another loop's opposite leg
        if ns[k] == m_new: m_new = m_old
        if ms[k] == n_new: n_new = n_old
    if n_new <= m_new + 1:
        return m_old, n_old
    return m_new, n_new
