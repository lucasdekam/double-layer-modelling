import constants as C
import numpy as np
from scipy.integrate import solve_bvp
from abc import ABC, abstractmethod
import os
import pickle
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class DoubleLayerModelSolution:
    x:      np.ndarray  # position, nm
    phi:    np.ndarray  # potential, V
    efield: np.ndarray  # electric field, V/m
    c_cat:  np.ndarray  # cation concentration, M
    c_an:   np.ndarray  # anion concentration, M
    c_sol:  np.ndarray  # solvent concentration, M
    eps:    np.ndarray  # relative permittivity

@dataclass
class PotentialSweepSolution:
    phi:    np.ndarray  # potential, V
    charge: np.ndarray  # surface charge, C/m^2
    cap:    np.ndarray  # differential capacity, uF/cm2


class DoubleLayerModel(ABC):
    """
    Abstract base class for a double layer model, requiring all 
    derived classes to have the same methods
    """
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def getPotentialSweepSolution(self, potential_V):
        pass

    @abstractmethod
    def getDoubleLayerModelSolution(self, xmax_m, N, phi0_V):
        pass


class GuyChapman(DoubleLayerModel):
    """
    Guy-Chapman model, treating ions as point particles obeying Boltzmann statistics.

    See e.g. Schmickler & Santos' Interfacial Electrochemistry.
    """
    def __init__(self, ionconc_M, pzc_V):
        self.n_0 = ionconc_M * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.z*C.e_0)**2/(C.eps_r_water*C.eps_0*C.k_B*C.T))
        self.pzc_V = pzc_V

    def getPotentialSweepSolution(self, potential_V):
        cap = self.kappa_debye * C.eps_r_water * C.eps_0 * np.cosh(C.beta*C.z*C.e_0/2 * (potential_V - self.pzc_V)) * 1e2
        ret = PotentialSweepSolution(phi=potential_V, charge=np.zeros(potential_V.shape), cap=cap)
        return ret

    def getDoubleLayerModelSolution(self, xmax_m, N, phi0_V):
        return None


class BorukhovAndelmanOrland(DoubleLayerModel):
    """
    Model developed by Borukhov, Andelman and Orland, modifying the Guy-Chapman model to 
    take finite ion size into account.

    https://doi.org/10.1016/S0013-4686(00)00576-4
    """
    def __init__(self, ionconc_M, pzc_V, d_ions_m):
        self.n_0 = ionconc_M * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.z*C.e_0)**2/(C.eps_r_water*C.eps_0*C.k_B*C.T))
        self.pzc_V = pzc_V

        self.a = d_ions_m
        self.chi0 = 2 * d_ions_m ** 3 * self.n_0 

    def getPotentialSweepSolution(self, potential_V):
        cap = np.sqrt(2) * self.kappa_debye * C.eps_r_water * C.eps_0 / np.sqrt(self.chi0) \
            * self.chi0 * np.sinh(C.beta*C.z*C.e_0*np.abs(potential_V)) \
            / (self.chi0 * np.cosh(C.beta*C.z*C.e_0*potential_V) - self.chi0 + 1) \
            / (2*np.sqrt(np.log(self.chi0 * np.cosh(C.beta*C.z*C.e_0*potential_V) - self.chi0 + 1))) \
            * 1e2 # uF/cm^2
        ret = PotentialSweepSolution(phi=potential_V, charge=np.zeros(potential_V.shape), cap=cap)
        return ret

    def odeSystem(self, x, y):
        """
        System of dimensionless 1st order ODE's that we solve

        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """        
        dy1 = y[1, :]
        dy2 = np.sinh(y[0, :]) / (1 - self.chi0 + self.chi0 * np.cosh(y[0, :]))
        
        return np.vstack([dy1, dy2])

    def bc(self, phi_bc_V):
        """
        Boundary conditions: fixed potential at the metal, zero potential at "infinity" (or: far enough away)

        Returns a boundary condition function for use in scipy's solve_bvp
        """
        return lambda ya, yb : np.array([ya[0] - phi_bc_V * C.beta * C.z * C.e_0, yb[0]])   

    def temp_pH_bc(self, pH_bulk):
        a_H_bulk = 10 ** (-pH_bulk)
        return lambda ya, yb : np.array([
            C.eps_r_water * C.eps_0 * ya[1] * self.kappa_debye / (C.beta * C.z * C.e_0) \
            - C.e_0 * C.N_sites * \
            (C.K_silica * (1 - self.chi0 + self.chi0 * np.cosh(ya[0])))/ \
            (C.K_silica * (1 - self.chi0 + self.chi0 * np.cosh(ya[0])) + a_H_bulk * np.exp(-ya[0])), 
            yb[0]
            ])   

    # def getXAxis_m(self, xmax_m, N):
    #     """
    #     Get a logarithmically spaced x-axis, fine mesh close to electrode
    #     """
    #     xmax_nm = xmax_m * 1e9
    #     expmax = np.log10(xmax_nm)
    #     x = np.logspace(-9, expmax, N) - 1e-9
    #     return x*1e-9

    def getOdeSol(self, xmax_m, N, phi0_V, verbose=False):
        """
        Solve the ODE system 
        """
        x = self.kappa_debye * np.linspace(0, xmax_m, N) # dimensionless x-axis
        y = np.zeros((2, x.shape[0])) 
        
        sol = solve_bvp(self.odeSystem, self.temp_pH_bc(7.0), x, y, max_nodes=1000000, verbose=verbose)

        return sol

    def getDoubleLayerModelSolution(self, xmax_m, N, phi0_V):
        sol = self.getOdeSol(xmax_m, N, phi0_V, verbose=True)
        BFc = np.exp(-sol.y[0, :])
        BFa = np.exp(sol.y[0, :])
        Omega = 1 - self.chi0 + self.chi0 * np.cosh(sol.y[0, :])

        c_cat = self.n_0 * BFc / Omega / C.N_A / 1e3
        c_an  = self.n_0 * BFa / Omega / C.N_A / 1e3
        c_sol = np.zeros(sol.x.shape)

        eps = np.ones(sol.x.shape) * C.eps_r_water
        ret = DoubleLayerModelSolution(
            x=sol.x / self.kappa_debye * 1e9, 
            phi=sol.y[0, :] / (C.beta * C.z * C.e_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.beta * C.z * C.e_0), 
            c_cat=c_cat,
            c_an=c_an,
            c_sol=c_sol,
            eps=eps)
        return ret


class Huang(DoubleLayerModel):
    """
    Model developed by Jun Huang and co-workers, taking into account finite ion size and 
    dipole moments of the solution molecules.

    https://doi.org/10.1021/acs.jctc.1c00098
    https://doi.org/10.1021/jacsau.1c00315
    """
    def __init__(self, ionconc_M, pzc_V, d_cation_m, d_anion_m, model_water_molecules=True):
        self.n_0 = ionconc_M * 1e3 * C.N_A
        self.pzc_V = pzc_V
        self.model_water_molecules = model_water_molecules

        n_water_bulk = C.c_water_bulk * C.N_A
        self.n_max = n_water_bulk + 2 * self.n_0 

        d_solvent_m = (1/self.n_max)**(1/3)  # water molecule diameter, m
        self.dc = d_cation_m 
        self.da = d_anion_m
        self.gamma_c = d_cation_m**3/d_solvent_m**3
        self.gamma_a = d_anion_m**3/d_solvent_m**3 
        self.chi = self.n_0 / self.n_max

        self.kappa = np.sqrt(2*self.n_0*(C.z*C.e_0)**2/(C.eps_0*C.eps_r_water*C.k_B*C.T))
        self.eps_r_opt = 1
        if not self.model_water_molecules:
            self.eps_r_opt = C.eps_r_water
        self.p = np.sqrt(3 * C.k_B * C.T * (C.eps_r_water - self.eps_r_opt) * C.eps_0 / n_water_bulk)
        self.ptilde = self.p * self.kappa / (C.z * C.e_0)

    def computeBoltzmannFactorsAndOmega(self, y):
        """
        Compute the Boltzmann factors
        BFc = exp(-z e beta phi)
        BFa = exp(+z e beta phi)
        BFs = sinh(beta p E)/(beta p E)

        minimum and maximum are to avoid infinities or division by zero
        """
        BFc = np.minimum(np.exp(-y[0, :]), 1e60)
        BFa = np.minimum(np.exp(+y[0, :]), 1e60)
        BFs = None
        if self.model_water_molecules:
            BFs = np.maximum(np.minimum(np.sinh(self.ptilde * y[1, :]), 1e60)/(self.ptilde * y[1, :] + 1e-60), 1e-60)
        else:
            BFs = 1 # If we don't model the water molecule dipoles, p=0 so sinh x/x = 1
        Omega = (1 - 2*self.chi) * BFs + self.gamma_c * self.chi * BFc + self.gamma_a * self.chi * BFa

        return BFc, BFa, BFs, Omega

    def langevinOfXOverX(self, x):
        """
        Returns L(x)/x, where L(x) is the Langevin function:
        L(x) = 1/tanh(x) - 1/x
        For small x, the function value is 1/3
        """
        ret = np.zeros(np.atleast_1d(x).shape)
        ret = (1/np.tanh(x) - 1/x)/x
        ret[np.abs(x) <= 1e-9] = 1/3
        return ret

    def computePermittivity(self, BFa, BFc, Omega, soly_1):
        """
        Compute the permittivity using the electric field
        """
        eps = None
        if self.model_water_molecules:
            eps = self.eps_r_opt + 1/2 * C.eps_r_water * self.ptilde**2 * (1 - self.chi * BFc / Omega - self.chi * BFa/Omega) * self.langevinOfXOverX(self.ptilde * soly_1) / self.chi
        else:
            eps = np.ones(soly_1.shape) * C.eps_r_water
        return eps
    
    def odeSystem(self, x, y):
        """
        System of dimensionless 1st order ODE's that we solve

        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """        
        dy1 = y[1, :]

        BFc, BFa, BFs, Omega = self.computeBoltzmannFactorsAndOmega(y)        
        H = 1/2 * (C.eps_r_water / self.eps_r_opt) * (1 - 1/BFs**2) * (1 - self.chi * BFc / Omega - self.chi * BFa/Omega) / self.chi

        dy2 = None
        if self.model_water_molecules:
            dy2 = - 1/2 * (C.eps_r_water / self.eps_r_opt) * (BFc - BFa) / Omega * y[1, :]**2 / (y[1, :]**2 + H + 1e-60)
        else:
            dy2 = - 1/2 * (C.eps_r_water / self.eps_r_opt) * (BFc - BFa) / Omega
        
        return np.vstack([dy1, dy2])

    def bc(self, phi_bc_V):
        """
        Boundary conditions: fixed potential at the metal, zero potential at "infinity" (or: far enough away)

        Returns a boundary condition function for use in scipy's solve_bvp
        """
        return lambda ya, yb : np.array([ya[0] - phi_bc_V * C.beta * C.z * C.e_0, yb[0]])   

    def getXAxis_m(self, xmax_m, N):
        """
        Get a logarithmically spaced x-axis, fine mesh close to electrode
        """
        xmax_nm = xmax_m * 1e9
        expmax = np.log10(xmax_nm)
        x = np.logspace(-9, expmax, N) - 1e-9
        return x*1e-9

    def getOdeSol(self, xmax_m, N, phi0_V, verbose=False, force_recalculation=False):
        """
        Solve the ODE system or load the solution if there is one (if we want to plot many things)
        """
        x = self.kappa * self.getXAxis_m(xmax_m, N) # dimensionless x-axis
        y = np.zeros((2, x.shape[0])) 

        sol = None 

        pickle_name = f'sol_huang__c0__{self.n_0/C.N_A/1e3:.3f}__xmax_{xmax_m*1e9:.0f}nm__N_{N}__phi0_{phi0_V:.2f}__dc_{self.dc*1e10:.0f}__da_{self.da*1e10:.0f}.pkl'
        folder_name = './solutions/'
        if pickle_name not in os.listdir(folder_name) or force_recalculation:
            sol = solve_bvp(self.odeSystem, self.bc(phi0_V), x, y, max_nodes=1000000, verbose=verbose)
            pickle.dump(sol, open(folder_name+pickle_name, 'wb'))
            if verbose:
                print(f'Solved and saved under {folder_name+pickle_name}.')
        else:
            if verbose:
                print(f'File {pickle_name} found.')
            sol = pickle.load(open(folder_name+pickle_name, 'rb'))

        return sol

    def getDoubleLayerModelSolution(self, xmax_m, N, phi0_V, force_recalculation=False):
        sol = self.getOdeSol(xmax_m, N, phi0_V, verbose=True, force_recalculation=force_recalculation)
        BFc, BFa, _, Omega = self.computeBoltzmannFactorsAndOmega(sol.y)
        
        c_cat = self.n_max * self.chi * BFc / Omega / C.N_A / 1e3
        c_an = self.n_max * self.chi * BFa / Omega / C.N_A / 1e3
        c_sol = self.n_max / C.N_A / 1e3 - c_cat - c_an

        eps = self.computePermittivity(BFa, BFc, Omega, sol.y[1, :])
        ret = DoubleLayerModelSolution(
            x=sol.x / self.kappa * 1e9, 
            phi=sol.y[0, :] / (C.beta * C.z * C.e_0),
            efield=-sol.y[1, :] * self.kappa / (C.beta * C.z * C.e_0), 
            c_cat=c_cat,
            c_an=c_an,
            c_sol=c_sol,
            eps=eps)
        return ret

    def computeCharge(self, xmax_m, N, potential_V, force_recalculation=False):
        """
        Compute the surface charge at a given potential
        """
        sol = self.getOdeSol(xmax_m, N, potential_V, verbose=False, force_recalculation=force_recalculation)   

        BFc, BFa, _, Omega = self.computeBoltzmannFactorsAndOmega(sol.y)
        eps = self.computePermittivity(BFa, BFc, Omega, sol.y[1, :])
        dphidx = - sol.y[1, :] * self.kappa / (C.beta * C.z * C.e_0)

        charge_C = C.eps_0 * eps[0] * dphidx[0]
        return charge_C

    def getPotentialSweepSolution(self, potential_V, force_recalculation=False):
        """
        Compute the differential capacitance in microfarads per square cm. 
        Parallelized using the multiprocessing Python module.
        """
        xmax_m = 100e-9
        N = 10000

        pool = mp.Pool(mp.cpu_count())
        charge = pool.starmap(self.computeCharge, [(xmax_m, N, phi, force_recalculation) for phi in potential_V])
        pool.close()

        cap = np.gradient(charge, edge_order=2) / np.gradient(potential_V) * 1e2
        ret = PotentialSweepSolution(phi=potential_V, charge=charge, cap=cap)
        return ret