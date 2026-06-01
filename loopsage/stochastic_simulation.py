#Basic Libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.stats as stats
from numba import njit, prange
from tqdm import tqdm
import importlib.resources

# scipy
from scipy.stats import norm
from scipy.stats import poisson

# My own libraries
from .preproc import *
from .plots import *
from .md import *
from .em import *

# Dynamically set the default path to the XML file in the package
try:
    with importlib.resources.path('loopsage.forcefields', 'classic_sm_ff.xml') as default_xml_path:
        default_xml_path = str(default_xml_path)
except FileNotFoundError:
    # If running in a development setup without the resource installed, fallback to a relative path
    default_xml_path = 'loopsage/forcefields/classic_sm_ff.xml'

@njit
def Kappa(mi,ni,mj,nj,cross_loop=True):
    '''
    Computes the crossing function of LoopSage.
    '''
    k=0.0
    if cross_loop:
        if mi<mj and mj<ni and ni<nj: k+=1 # np.abs(ni-mj)+1
        if mj<mi and mi<nj and nj<ni: k+=1 # np.abs(nj-mi)+1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

@njit
def E_bind(L, R, ms, ns, bind_norm):
    '''
    The binding energy.
    '''
    binding = np.sum(L[ms] + R[ns])
    E_b = bind_norm * binding
    return E_b

@njit
def E_cross(ms, ns, k_norm, N_lef, cross_loop=True, between_families_penalty=True):
    '''
    The crossing energy.
    '''
    crossing = 0.0
    for i in prange(len(ms)):
        for j in range(i + 1, len(ms)):
            if between_families_penalty or (i < N_lef and j < N_lef) or (i >= N_lef and j >= N_lef):
                crossing += Kappa(ms[i], ns[i], ms[j], ns[j], cross_loop)
    return k_norm * crossing

@njit
def E_fold(ms, ns, fold_norm):
    ''''
    The folding energy.
    '''
    folding = np.sum(np.log(ns - ms))
    return fold_norm * folding

@njit
def E_bw(N_bws, r, BWs, ms, ns):
    '''
    Calculation of the RNApII binding energy. Needs cohesins positions as input.
    '''
    E_bw = 0
    for i in range(N_bws):
        E_bw += r[i] * np.sum(BWs[i, ms] + BWs[i, ns]) / np.sum(BWs[i])
    return E_bw

@njit
def E_epi(S, J, h, epi_norm):
    '''
    Potts model energy:
    - sparse pairwise J (zeros ignored)
    - optional external field h (can be None)
    '''

    N_beads = len(S)
    E = 0.0

    # Pairwise term
    for i in range(N_beads):
        Si = S[i]
        for j in range(i + 1, N_beads):
            Jij = J[i, j]
            if Jij != 0.0:
                if S[j] == Si:
                    E -= Jij

    # External field (optional)
    if h is not None:
        for i in range(N_beads):
            E -= h[i] * S[i]

    return epi_norm * E

@njit
def get_E(L, R, bind_norm, fold_norm, fold_norm2, k_norm,
          ms, ns, N_lef, N_lef2,
          cross_loop,
          r=None, N_bws=0, BWs=None,
          between_families_penalty=True,
          S=None, J=None, h=None, epi_norm=0.0):
    '''
    Total energy including optional Potts epigenetic term.
    '''

    energy = (
        E_bind(L, R, ms, ns, bind_norm)
        + E_cross(ms, ns, k_norm, cross_loop, between_families_penalty)
        + E_fold(ms, ns, fold_norm)
    )

    # second LEF family folding
    if fold_norm2 != 0:
        energy += E_fold(ms[N_lef:N_lef+N_lef2],
                         ns[N_lef:N_lef+N_lef2],
                         fold_norm2)

    # binding tracks
    if r is not None and BWs is not None:
        energy += E_bw(N_bws, r, BWs, ms, ns)

    # Potts epigenetic energy
    if epi_norm != 0.0 and J is not None and h is not None and S is not None:
        energy += E_epi(S, J, h, epi_norm)

    return energy

@njit
def get_dE_bind(L,R,bind_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for binding energy.
    '''
    return bind_norm*(L[m_new]+R[n_new]-L[ms[idx]]-R[ns[idx]])
    
@njit
def get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for folding energy.
    '''
    return fold_norm*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))

@njit
def get_dE_bw(N_bws, r, BWs, ms, ns, m_new, n_new, idx):
    dE_bw = 0
    for i in range(N_bws):
        dE_bw += r[i] * (BWs[i, m_new] + BWs[i, n_new] - BWs[i, ms[idx]] - BWs[i, ns[idx]]) / np.sum(BWs[i])
    return dE_bw

@njit
def get_dE_epi(S, J, h, epi_norm, k, s_new):
    '''
    Energy difference for Potts model when updating a single site k:
    S[k] -> s_new

    Supports:
    - sparse J
    - optional field h (can be None)
    '''
    
    s_old = S[k]

    # no change -> no cost
    if s_old == s_new:
        return 0.0

    dE = 0.0
    N_beads = len(S)

    # Pairwise interaction term
    for j in range(N_beads):
        if j == k:
            continue
        Jij = J[k, j]
        # skip zeros (critical for speed)
        if Jij != 0.0:
            Sj = S[j]
            # remove old contribution
            if Sj == s_old:
                dE += Jij
            # add new contribution
            if Sj == s_new:
                dE -= Jij
    
    # External field term (optional)
    if h is not None:
        dE += h[k] * (s_old - s_new)

    return epi_norm * dE

@njit
def get_dE_cross(ms, ns, m_new, n_new, idx, k_norm, cross_loop, N_lef, between_families_penalty):
    '''
    Energy difference for crossing energy.
    '''
    K1, K2 = 0, 0
    for i in prange(len(ms)):
        if i != idx:
            if between_families_penalty or (idx < N_lef and i < N_lef) or (idx >= N_lef and i >= N_lef):
                K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i], cross_loop)
                K2 += Kappa(m_new, n_new, ms[i], ns[i], cross_loop)
    return k_norm * (K2 - K1)

@njit
def get_dE_edges(L, R, bind_norm, fold_norm, fold_norm2, k_norm,
                 ms, ns, m_new, n_new, idx,
                 N_lef, N_lef2, cross_loop,
                 r=None, N_bws=0, BWs=None,
                 between_families_penalty=True):
    '''
    Energy difference for LEF (edge) updates only.
    '''

    dE = 0.0

    # Folding
    if idx < N_lef:
        dE += get_dE_fold(fold_norm,
                         ms[:N_lef], ns[:N_lef],
                         m_new, n_new, idx)
    else:
        dE += get_dE_fold(fold_norm2,
                         ms[N_lef:N_lef+N_lef2],
                         ns[N_lef:N_lef+N_lef2],
                         m_new, n_new, idx - N_lef)

    # Binding
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)

    # Crossing
    dE += get_dE_cross(ms, ns, m_new, n_new, idx,
                       k_norm, cross_loop, N_lef,
                       between_families_penalty)

    # Optional BW term
    if r is not None and BWs is not None:
        dE += get_dE_bw(N_bws, r, BWs, ms, ns, m_new, n_new, idx)

    return dE

@njit
def get_dE_nodes(S, J, h, epi_norm, k_spin, s_new):
    '''
    Energy difference for a single spin (node) update.
    '''

    if epi_norm == 0.0 or J is None or S is None:
        return 0.0

    return get_dE_epi(S, J, h, epi_norm, k_spin, s_new)

@njit
def has_cross(m, n, ms, ns, N_lef):
    """
    Checks if a candidate LEF (m, n) crosses any existing LEF.
    Only scans relevant LEFs.
    """
    for i in range(N_lef):
        mi = ms[i]
        ni = ns[i]

        # ignore identical index in updates externally if needed

        # crossing condition
        if (mi < m and m < ni and n > ni) or (m < mi and mi < n and ni > n):
            return True

    return False

@njit
def unbind_bind(N_beads, track, ms, ns, N_lef, enforce_no_cross=False):
    '''
    Rebinding Monte-Carlo step with optional crossing constraint.
    '''

    max_tries = 20  # small fixed budget for speed safety

    for _ in range(max_tries):
        # propose m_new
        if track is not None:
            weights = track / np.sum(track)
            m_new = np.searchsorted(np.cumsum(weights), np.random.rand())
        else:
            m_new = np.random.randint(0, N_beads - 3)

        n_new = m_new + 2

        # enforce non-crossing
        if enforce_no_cross:
            if not has_cross(m_new, n_new, ms, ns, N_lef):
                return int(m_new), int(n_new)
        else:
            return int(m_new), int(n_new)

    # fallback (if stuck, return old-style safe move)
    return int(m_new), int(n_new)

@njit
def slide(m_old, n_old, ms, ns, N_beads, rw=True, drift=True, enforce_no_cross=False, idx=0, N_lef=0):
    '''
    Sliding Monte-Carlo step with optional crossing constraint.
    '''

    choices = np.array([-1, 1], dtype=np.int64)

    r1 = np.random.choice(choices) if rw else -1
    r2 = np.random.choice(choices) if rw else 1

    m_new = max(m_old + r1, 0)
    if np.any(ns == m_new) and drift and m_old - r1 < n_old - 1:
        m_new = max(m_old - r1, 0)

    n_new = min(n_old + r2, N_beads - 1)
    if np.any(ms == n_new) and drift and n_old - r2 > m_old + 1:
        n_new = min(n_old - r2, N_beads - 1)

    # crossing constraint check
    if enforce_no_cross:
        if has_cross(m_new, n_new, ms, ns, N_lef):
            return int(m_old), int(n_old)

    return int(m_new), int(n_new)

@njit
def unfolding_metric(ms,ns,N_beads):
    '''
    This is a metric for the number of gaps (regions unfolded that are not within a loop).
    Cohesin positions are needed as input.
    '''
    fiber = np.zeros(N_beads)
    for i in range(len(ms)):
        fiber[ms[i]:ns[i]]=1
    unfold = 2*(N_beads-np.count_nonzero(fiber))/N_beads
    return unfold

@njit
def initialize(N_beads, N_lef, track, N_epi_states, S_mode_random=True):
    """
    Numba-safe initialization of LEFs + epigenetic states.

    Notes
    -----
    - S_mode_random = True -> random states
    - S_mode_random = False -> uniform state (0)
    """

    # LEF arrays
    ms = np.zeros(N_lef, dtype=np.int64)
    ns = np.zeros(N_lef, dtype=np.int64)

    for i in range(N_lef):
        m, n = unbind_bind(N_beads, track, ms, ns, N_lef)
        ms[i] = m
        ns[i] = n

    # Epigenetic states
    S = np.zeros(N_beads, dtype=np.int64)

    if S_mode_random:
        for i in range(N_beads):
            S[i] = np.random.randint(0, N_epi_states)
    else:
        pass

    return ms, ns, S

@njit
def run_simulation(N_beads, N_steps, MC_step, burnin,
                   T, T_min,
                   fold_norm, fold_norm2, bind_norm, k_norm,
                   N_lef, N_lef2,
                   L, R,
                   mode,
                   lef_rw=True, lef_drift=True,
                   cross_loop=True,
                   r=None, N_bws=0, BWs=None,
                   track=None,
                   between_families_penalty=True,
                   J=None, h=None, epi_norm=0.0,
                   N_epi_states=3,
                   p_spin=0.5):

    '''
    Runs the Monte Carlo simulation with LEF + Potts dynamics.
    '''

    Ti = T
    bi = burnin // MC_step

    # NEW: decide if spin moves are allowed
    spin_allowed = (J is not None) and (epi_norm != 0.0)

    # Initialization
    ms, ns, S = initialize(N_beads, N_lef + N_lef2, track, N_epi_states, True)

    E = get_E(L, R, bind_norm, fold_norm, fold_norm2, k_norm,
              ms, ns, N_lef, N_lef2, cross_loop,
              r, N_bws, BWs, between_families_penalty,
              S, J, h, epi_norm)

    # Storage
    n_save = N_steps // MC_step

    Es = np.zeros(n_save, dtype=np.float64)
    Ks = np.zeros(n_save, dtype=np.float64)
    Fs = np.zeros(n_save, dtype=np.float64)
    Bs = np.zeros(n_save, dtype=np.float64)
    ufs = np.zeros(n_save, dtype=np.float64)

    Ms = np.zeros((N_lef + N_lef2, n_save), dtype=np.int64)
    Ns = np.zeros((N_lef + N_lef2, n_save), dtype=np.int64)

    epi_states = np.zeros((N_beads, n_save), dtype=np.int64)

    last_percent = -1

    # NEW: total MC proposals per step
    N_swift = 2 * (N_lef + N_lef2)

    # MAIN LOOP
    for i in range(N_steps):

        percent = int(100 * i / N_steps)
        if percent % 5 == 0 and percent != last_percent:
            print(f"Progress: {percent} % completed.")
            last_percent = percent

        Ti = T - (T - T_min) * (i + 1) / N_steps if mode == 'Annealing' else T

        # NEW MC SCHEME: N_swift proposals per step
        for _ in range(N_swift):

            do_spin = spin_allowed and (np.random.rand() < p_spin)

            if do_spin:
                # SPIN MOVE
                k = np.random.randint(0, N_beads)
                s_old = S[k]

                s_new = np.random.randint(0, N_epi_states)
                if s_new == s_old:
                    s_new = (s_old + 1) % N_epi_states

                dE = get_dE_nodes(S, J, h, epi_norm, k, s_new)

                if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                    S[k] = s_new
                    E += dE

            else:
                # LEF MOVE (single LEF updated per proposal)
                j = np.random.randint(0, N_lef + N_lef2)

                r_move = np.random.randint(0, 2)

                if r_move == 0:
                    m_new, n_new = unbind_bind(N_beads, track, ms, ns, N_lef)
                else:
                    m_new, n_new = slide(ms[j], ns[j], ms, ns,
                                         N_beads, lef_rw, lef_drift)

                dE = get_dE_edges(L, R, bind_norm,
                                  fold_norm, fold_norm2, k_norm,
                                  ms, ns, m_new, n_new, j,
                                  N_lef, N_lef2,
                                  cross_loop,
                                  r, N_bws, BWs,
                                  between_families_penalty)

                if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                    ms[j], ns[j] = m_new, n_new
                    E += dE

        # SAVE STATE (unchanged logic)
        if i % MC_step == 0:

            idx_save = i // MC_step

            ufs[idx_save] = unfolding_metric(ms, ns, N_beads)
            Es[idx_save] = E
            Ks[idx_save] = E_cross(ms, ns, k_norm, cross_loop,
                                   N_lef, between_families_penalty)
            Fs[idx_save] = E_fold(ms, ns, fold_norm)
            Bs[idx_save] = E_bind(L, R, ms, ns, bind_norm)

            for b in range(N_beads):
                epi_states[b, idx_save] = S[b]

            for j in range(N_lef + N_lef2):
                Ms[j, idx_save] = ms[j]
                Ns[j, idx_save] = ns[j]

    return Ms, Ns, Es, Ks, Fs, Bs, ufs, epi_states

class StochasticSimulation:
    def __init__(
            self,
            bedpe_file,
            chrom,
            region=None,
            N_beads=None,
            N_lef=None,
            N_lef2=0,
            out_dir=None,
            bw_files=None,
            lef_density_file=None,
            comp_file=None
        ):
            """
            Chromatin stochastic simulation initializer.

            Parameters
            ----------
            region : list[int, int]
                Genomic region [start, end] in base pairs.

            chrom : str
                Chromosome name (e.g. "chr1").

            bedpe_file : str
                BEDPE file containing CTCF/loop interactions.

            N_beads : int or None
                Number of polymer beads. If None, inferred from region at ~2kb resolution.

            N_lef : int or None
                Number of loop extrusion factors (LEFs). If None, inferred from CTCF count.

            N_lef2 : int
                Second LEF population size (optional heterogeneous population).

            out_dir : str or None
                Output directory for simulation results.

            bw_files : list[str] or None
                BigWig signal tracks (e.g. ChIP-seq, compartments).

            lef_density_file : str or None
                Optional track defining spatial LEF loading probability.

            comp_file : str or None
                Compartment track (BigWig or BED format).
            """
            # Basic geometry
            self.chrom = chrom
            self.region = region
            self.resolve_region()

            self.N_beads = (
                N_beads
                if N_beads is not None
                else int(np.round((region[1] - region[0]) / 2000))
            )

            # Input data
            self.bedpe_file = bedpe_file
            self.bw_files = bw_files
            self.lef_density_file = lef_density_file
            self.comp_file = comp_file

            self.N_bws = len(bw_files) if bw_files is not None else 0
            self.path = make_folder(out_dir)

            # Print basic setup
            print(f"Number of beads: {self.N_beads}")
            self.preprocessing()

            # LEF initialization
            self.N_lef = (
                self.N_CTCF//2
                if N_lef is None
                else N_lef
            )
            self.N_lef2 = N_lef2

            print(f"Number of LEFs: {self.N_lef + self.N_lef2}")

    def resolve_region(self):

        chrom_len = CHROM_LENGTHS.get(self.chrom, None)
        if chrom_len is None:
            raise ValueError(f"Unknown chromosome: {self.chrom}")

        region = self.region
        used_fallback = False

        if (
            region is None
            or not isinstance(region, (list, tuple))
            or len(region) != 2
        ):
            region = [0, chrom_len]
            used_fallback = True
        else:
            start, end = region

            try:
                start = int(start)
                end = int(end)
                region = [start, end]
            except Exception:
                region = [0, chrom_len]
                used_fallback = True

            if region[1] <= region[0]:
                region = [0, chrom_len]
                used_fallback = True
        
        self.region = region

        tag = "FALLBACK" if used_fallback else "OK"
        print(f"[resolve_region:{tag}] chrom={self.chrom}, region={self.region}, length={self.region[1] - self.region[0]}")
    
    def run_energy_minimization(self, N_steps, MC_step, burnin, T=1, T_min=0, mode='Metropolis', viz=False, save=False, f=1.0, f2=0.0, b=1.0, kappa=1.0, epi_coeff=0.0, N_epi_states=3, p_spin=0.5, lef_rw=True, lef_drift=True, cross_loop=True, r=None, between_families_penalty=True):
        '''
        Implementation of the stochastic Monte Carlo simulation.

        Input parameters:
        N_steps (int): number of Monte Carlo steps.
        MC_step (int): sampling frequency.
        burnin (int): definition of the burnin period.
        T (float): simulation (initial) temperature.
        mode (str): it can be either 'Metropolis' or 'Annealing'.
        viz (bool): True in case that user wants to see plots.
        r (list): strength of each ChIP-Seq experiment.
        N_bws (int): number of binding weight matrices.
        BWs (np.ndarray): binding weight matrices.
        between_families_penalty (bool): whether to apply penalty for interactions between families.
        '''
        # Define normalization constants
        N_lef_tot = self.N_lef + self.N_lef2
        fold_norm = -2*(2-np.log(self.avg_length)/np.log(self.max_length))*f#-self.N_beads * f  / (N_lef_tot * log_term )
        print("Folding coefficient after normalization:",fold_norm)
        fold_norm2 = -2*(2-np.log(self.avg_length)/np.log(self.max_length))*f2
        bind_norm = -self.N_beads * b / (np.sum(self.L) + np.sum(self.R))
        print("Binding coefficient after normalization:",bind_norm)
        k_norm = kappa * 1e6
        # epi_scale = self.N_beads + 0.5 * self.N_CTCF
        epi_norm = epi_coeff
        self.N_steps, self.MC_step = N_steps, MC_step
        r = np.full(self.N_bws, -self.N_beads / 10) if not r and self.N_bws > 0 else (None if not r else r)

        # Run simulation
        print('\nRunning simulation (with numba acceleration)...')
        start = time.time()
        self.burnin = burnin
        self.Ms, self.Ns, self.Es, self.Ks, self.Fs, self.Bs, self.ufs, self.epi_states = run_simulation(
            self.N_beads,
            N_steps,
            MC_step,
            burnin,
            T,
            T_min,
            fold_norm,
            fold_norm2,
            bind_norm,
            k_norm,
            self.N_lef,
            self.N_lef2,
            self.L,
            self.R,
            mode,
            lef_rw,
            lef_drift,
            cross_loop,
            r,
            self.N_bws,
            self.BWs,
            self.lef_track,
            between_families_penalty,
            J=self.J,                 # NEW
            h=self.h,                 # NEW
            epi_norm=epi_norm,   # NEW
            N_epi_states=N_epi_states,   # NEW
            p_spin=p_spin        # NEW
        )        
        end = time.time()
        elapsed = end - start
        print(f'Computation finished successfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and {elapsed%60:.0f} seconds.')
        
        # Save simulation info
        if save:
            save_dir = os.path.join(self.path, 'other') + '/'
            with open(save_dir + 'info.txt', "w") as file:
                file.write(f'Number of beads {self.N_beads}.\n')
                file.write(f'Number of cohesins {self.N_lef}. Number of cohesins in second family {self.N_lef2}. Number of CTCFs {self.N_CTCF}. \n')
                file.write(f'Bedpe file for CTCF binding is {self.bedpe_file}.\n')
                file.write(f'Initial temperature {T}. Minimum temperature {T_min}.\n')
                file.write(f'Monte Carlo optimization method: {mode}.\n')
                file.write(f'Monte Carlo steps {N_steps}. Sampling frequency {self.MC_step}. Burnin period {burnin}.\n')
                file.write(f'Crossing energy in equilibrium is {np.average(self.Ks[burnin//MC_step:]):.2f}. Crossing coefficient kappa={kappa}.\n')
                file.write(f'Folding energy in equilibrium is {np.average(self.Fs[burnin//MC_step:]):.2f}. Folding coefficient f={f}. Folding coefficient for the second family f2={f2}\n')
                file.write(f'Binding energy in equilibrium is {np.average(self.Bs[burnin//MC_step:]):.2f}. Binding coefficient b={b}.\n')
                if r is not None and self.BWs is not None:
                    file.write(f'RNApII binding energy included with {self.N_bws} binding weight matrices.\n')
                file.write(f'Energy at equilibrium: {np.average(self.Es[self.burnin//MC_step:]):.2f}.\n')
            np.save(save_dir + 'Ms.npy', self.Ms)
            np.save(save_dir + 'Ns.npy', self.Ns)
            np.save(save_dir + 'ufs.npy', self.ufs)
            np.save(save_dir + 'Es.npy', self.Es)
            np.save(save_dir + 'Bs.npy', self.Bs)
            np.save(save_dir + 'Fs.npy', self.Fs)
            np.save(save_dir + 'Ks.npy', self.Ks)
        
        # Some visualizations
        if viz:
            coh_traj_plot(self.Ms, self.Ns, self.N_beads, self.path)
            plot_epi_trajectory(self.epi_states, self.path)
            make_timeplots(self.Es, self.Bs, self.Ks, self.Fs, burnin//MC_step, mode, self.path)
            coh_probdist_plot(self.Ms, self.Ns, self.N_beads, self.path)
            stochastic_heatmap(self.Ms, self.Ns, self.N_beads, self.path, method='tanh')
        
        return self.Es, self.Ms, self.Ns, self.Bs, self.Ks, self.Fs, self.ufs, self.epi_states

    def preprocessing(self):
        """
        Preprocessing pipeline using updated BEDPE + BigWig exporters.

        Produces:
        - L, R, J interaction structures
        - BW tracks (compartments, ChIP, etc.)
        - LEF track
        - basic dataset statistics
        """

        # 1. Chromatin structure from BEDPE
        L, R, J, J_loss, stats = binding_vectors_from_bedpe(
            bedpe_file=self.bedpe_file,
            N_beads=self.N_beads,
            region=self.region,
            chrom=self.chrom,
            out_path=self.path,
            normalization=False,
            viz=True,
            diagonal_interactions=True,
            alpha=1.0,
            smooth=True,
            smooth_sigma=self.N_beads/200
        )
        
        self.L = L
        self.R = R
        self.J, self.J_loss = J, J_loss
        self.loop_stats = stats

        # 2. CTCF / loop count estimate
        self.N_CTCF = int(stats["n_loops"])
        self.avg_length = int(stats["loop_length"]["mean"])
        self.max_length = int(stats["loop_length"]["max"])
        print("Number of CTCF:", self.N_CTCF)

        # 3. BigWig tracks (compartments, ChIP, etc.)
        if not self.bw_files:
            self.BWs = None
            self.N_bws = 0

        else:
            if isinstance(self.bw_files, str):
                self.bw_files = [self.bw_files]

            self.N_bws = len(self.bw_files)
            self.BWs = np.zeros((self.N_bws, self.N_beads), dtype=np.float64)

            for i, f in enumerate(self.bw_files):
                exporter = BWExporter(
                    path=f,
                    region=self.region,
                    chrom=self.chrom,
                    N_beads=self.N_beads
                )

                self.BWs[i, :] = exporter.load_track(
                    viz=True,
                    roll=False,
                    norm=None,
                    out_path=self.path,
                    scale_minus1_1=False
                )

        # 4. LEF / epigenetic track
        if self.lef_density_file:
            exporter = BWExporter(
                path=self.lef_density_file,
                region=self.region,
                chrom=self.chrom,
                N_beads=self.N_beads
            )

            self.lef_track = exporter.load_track(
                viz=True,
                roll=True,
                norm=None,
                out_path=self.path,
                scale_minus1_1=False
            )
        else:
            self.lef_track = None
        
        # 5. Compartments -> continuous external field h
        if self.comp_file:

            f = self.comp_file.lower()

            # CASE 1: BigWig
            if f.endswith((".bw", ".bigwig")):

                exporter = BWExporter(
                    path=self.comp_file,
                    region=self.region,
                    chrom=self.chrom,
                    N_beads=self.N_beads
                )

                self.h = exporter.load_track(
                    viz=True,
                    roll=False,
                    norm=None,
                    scale_minus1_1=True
                )

            # CASE 2: BED-like compartments
            elif f.endswith((".bed", ".bed.gz")):

                self.h = load_compartments_bed(
                    bed_file=self.comp_file,
                    region=self.region,
                    out_path=self.path,
                    chrom=self.chrom,
                    N_beads=self.N_beads,
                    use_score=True,
                    spline_smooth=True,
                    spline_s=self.N_beads/100,
                    scale_minus1_1=True,
                    viz=True,
                    debug=False
                )

            # fallback
            else:
                raise ValueError(
                    f"Unsupported compartment format: {self.comp_file}"
                )

            self.h = np.asarray(self.h, dtype=np.float64)

        else:
            self.h = None

    def run_EM(self,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=100.0,ev_ff_power=3.0,tolerance=0.001,friction=0.1,integrator_step=100*mm.unit.femtosecond,temperature=310,init_struct='rw',save_plots=True,ff_path=default_xml_path):
        em = EM_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,ev_ff_power,tolerance)
        sim_heat = em.run_pipeline(plots=save_plots,friction=friction,integrator_step=integrator_step,temperature=temperature,ff_path=ff_path,init_struct=init_struct)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)
    
    def run_MD(self,platform='CPU',angle_ff_strength=200,le_distance=0.1,le_ff_strength=50000.0,ev_ff_strength=100.0,ev_ff_power=3.0,do_compartments=False,tolerance=0.001,friction=0.1,integrator_step=100*mm.unit.femtosecond,temperature=310,init_struct='rw',sim_step=1000,save_plots=True,ff_path=default_xml_path,p_ev=0,continuous_topoisomerase=False):
        if not do_compartments: self.epi_states=None
        md = MD_LE(self.Ms,self.Ns,self.epi_states,self.N_beads,self.path,platform,angle_ff_strength,le_distance,le_ff_strength,ev_ff_strength,ev_ff_power,tolerance)
        sim_heat = md.run_pipeline(plots=save_plots,sim_step=sim_step,friction=friction,integrator_step=integrator_step,temperature=temperature,ff_path=ff_path,p_ev=p_ev,init_struct=init_struct,continuous_topoisomerase=continuous_topoisomerase)
        corr_exp_heat(sim_heat,self.bedpe_file,self.region,self.chrom,self.N_beads,self.path)

def main():
    # Definition of Monte Carlo parameters
    N_steps, MC_step, burnin, T, T_min = int(4e4), int(5e2), 1000, 3.0, 1.0
    N_lef, N_lef2 = 100, 20
    lew_rw = True
    mode = 'Annealing'
    
    # Simulation Strengths
    f, f2, b, kappa = 1.0, 2.0, 1.0, 1.0
    
    # Definition of region
    region, chrom = [15550000, 16850000], 'chr6'
    
    # Definition of data
    output_name = 'tmp'
    bedpe_file = '/home/skorsak/Data/HiChIP/Maps/hg00731_smc1_maps_2.bedpe'
    
    # Between family penalty
    between_families_penalty = True
    
    sim = StochasticSimulation(region, chrom, bedpe_file, out_dir=output_name, N_beads=1000, N_lef=N_lef, N_lef2=N_lef2)
    Es, Ms, Ns, Bs, Ks, Fs, ufs = sim.run_energy_minimization(
        N_steps, MC_step, burnin, T, T_min, mode=mode, viz=True, save=True, f=f, f2=f2, b=b, kappa=kappa, lef_rw=lew_rw, between_families_penalty=between_families_penalty
    )
    sim.run_MD('CUDA', continuous_topoisomerase=True, p_ev=0.01)

if __name__=='__main__':
    main()