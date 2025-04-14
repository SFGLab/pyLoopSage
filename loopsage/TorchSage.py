import torch

class StochasticSimulationTorch:
    def __init__(self, N_beads, N_lef, N_lef2=0, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.N_beads = N_beads
        self.N_lef = N_lef
        self.N_lef2 = N_lef2
        self.N_total = N_lef + N_lef2
        self.ms, self.ns = self.initialize()
    
    def initialize(self):
        ms = torch.randint(0, self.N_beads - 2, (self.N_total,), device=self.device)
        ns = ms + 2
        return ms, ns
    
    def kappa(self, mi, ni, mj, nj, cross_loop=True):
        k = torch.zeros_like(mi, dtype=torch.float32)
        if cross_loop:
            k += ((mi < mj) & (mj < ni) & (ni < nj)).float()
            k += ((mj < mi) & (mi < nj) & (nj < ni)).float()
        k += ((mj == ni) | (mi == nj) | (ni == nj) | (mi == mj)).float()
        return k

    def e_bind(self, L, R, ms, ns, bind_norm):
        binding = torch.sum(L[ms] + R[ns])
        return bind_norm * binding

    def e_cross(self, ms, ns, k_norm, cross_loop=True):
        crossing = 0.0
        for i in range(len(ms)):
            mi, ni = ms[i], ns[i]
            mj, nj = ms[i+1:], ns[i+1:]
            crossing += torch.sum(self.kappa(mi, ni, mj, nj, cross_loop))
        return k_norm * crossing

    def e_fold(self, ms, ns, fold_norm):
        folding = torch.sum(torch.log(ns - ms))
        return fold_norm * folding

    def total_energy(self, L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms, ns, cross_loop=True):
        energy = self.e_bind(L, R, ms, ns, bind_norm) + self.e_cross(ms, ns, k_norm, cross_loop) + self.e_fold(ms, ns, fold_norm)
        if fold_norm2 != 0:
            energy += self.e_fold(ms[self.N_lef:], ns[self.N_lef:], fold_norm2)
        return energy

    def mc_step(self, L, R, bind_norm, fold_norm, fold_norm2, k_norm, T, cross_loop=True):
        idx = torch.randint(0, self.N_total, (1,), device=self.device).item()
        move_type = torch.randint(0, 2, (1,), device=self.device).item()
        m_old, n_old = self.ms[idx], self.ns[idx]
        
        if move_type == 0:
            m_new = torch.randint(0, self.N_beads - 2, (1,), device=self.device).item()
            n_new = m_new + 2
        else:
            shift = torch.randint(-1, 2, (1,), device=self.device).item()
            m_new = torch.clamp(m_old + shift, 0, self.N_beads - 2)
            n_new = torch.clamp(n_old + shift, 2, self.N_beads - 1)
        
        ms_new = self.ms.clone()
        ns_new = self.ns.clone()
        ms_new[idx] = m_new
        ns_new[idx] = n_new
        
        E_old = self.total_energy(L, R, bind_norm, fold_norm, fold_norm2, k_norm, self.ms, self.ns, cross_loop)
        E_new = self.total_energy(L, R, bind_norm, fold_norm, fold_norm2, k_norm, ms_new, ns_new, cross_loop)
        dE = E_new - E_old
        
        if dE <= 0 or torch.exp(-dE / T) > torch.rand(1, device=self.device):
            self.ms[idx] = m_new
            self.ns[idx] = n_new

    def run_simulation(self, L, R, N_steps, T, bind_norm, fold_norm, fold_norm2, k_norm, cross_loop=True):
        energies = []
        for step in tqdm(range(N_steps)):
            self.mc_step(L, R, bind_norm, fold_norm, fold_norm2, k_norm, T, cross_loop)
            if step % 100 == 0:
                E = self.total_energy(L, R, bind_norm, fold_norm, fold_norm2, k_norm, self.ms, self.ns, cross_loop)
                energies.append(E.item())
        return energies