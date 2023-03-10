"""
Implementation of double-layer models
"""
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

import constants as C
import langevin as L


@dataclass
class SpatialProfilesSolution:
    """
    Dataclass to store double-layer model solution data
    x: position, nm
    phi: potential, V
    efield: electric field, V/m
    c_dict: dict of concentration profiles of form {'name of species': np.ndarray}
    eps: relative permittivity
    name: name of the model
    """
    x:       np.ndarray # pylint: disable=invalid-name
    phi:     np.ndarray
    efield:  np.ndarray
    c_dict:  dict
    eps:     np.ndarray
    name:    str


@dataclass
class PotentialSweepSolution:
    """
    Data class to store potential sweep solution data, e.g. differential capacitance
    phi: potential range, V
    charge: calculated surface charge, C/m^2
    cap: differential capacitance, uF/cm^2
    """
    phi:    np.ndarray
    charge: np.ndarray
    cap:    np.ndarray
    c_surf: dict
    efield: np.ndarray
    name:   str

@dataclass
class PhSweepSolution:
    """
    Data class to store potential sweep solution data, e.g. differential capacitance
    phi: potential range, V
    charge: calculated surface charge, C/m^2
    cap: differential capacitance, uF/cm^2
    """
    phi:    np.ndarray
    charge: np.ndarray
    cap:    np.ndarray
    c_h:    np.ndarray
    name:   str

class DoubleLayerModel:
    """
    Base class for an ODE. Implements basic features, but leaves
    RHS of ODE and the definition of number densities abstract.
    """
    def __init__(self, ion_concentration_molar: float) -> None:
        self.c_0 = ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.kappa_debye = np.sqrt(2*self.n_0*(C.Z*C.E_0)**2 /
                                   (C.EPS_R_WATER*C.EPS_0*C.K_B*C.T))
        self.name = 'Unnamed'

    def create_x_mesh(self, xmax_nm, n_points):
        """
        Get a logarithmically spaced x-axis, fine mesh close to electrode
        """
        max_exponent = np.log10(xmax_nm)
        x_nm = np.logspace(-6, max_exponent, n_points) - 1e-6
        return self.kappa_debye * 1e-9 * x_nm

    @abstractmethod
    def ode_rhs(self, x, y):  # pylint: disable=unused-argument, invalid-name
        """
        Function to pass to ODE solver, specifying the right-hand side of the
        system of dimensionless 1st order ODE's that we solve.
        x: dimensionless x-axis of length n, i.e. kappa (1/m) times x-position (m).
        y: dimensionless potential phi and dphi/dx, size 2 x n.
        """

    def dirichlet(self, phi0):
        """
        Return a boundary condition function to pass to scipy's solve_bvp
        """
        return lambda ya, yb: np.array([ya[0] - C.BETA * C.Z * C.E_0 * phi0, yb[0]])

    def odesolve_dirichlet(self,
            x_axis: np.ndarray,
            y_initial: np.ndarray,
            phi0: float,
            tol: float=1e-3):
        """
        Wrapper for scipy's solve_bvp, using the class methods
        """
        sol = solve_bvp(
            self.ode_rhs,
            self.dirichlet(phi0),
            x_axis,
            y_initial,
            tol=tol,
            max_nodes=int(1e8),
            verbose=0)
        return sol

    def sequential_solve(self, potential: np.ndarray, tol: float=1e-3):
        """
        Sweep over a potential array and use the previous solution as initial
        condition for the next.

        Returns: charge for each potential; last solution
        """
        chg = np.zeros(potential.shape)
        efield = np.zeros(potential.shape)
        c_surf = {'Cations': np.zeros(potential.shape),
                  'Anions': np.zeros(potential.shape),
                  'Solvent': np.zeros(potential.shape)}
        max_res = np.zeros(potential.shape)

        x_axis = self.create_x_mesh(100, 1000)
        y_initial = np.zeros((2, x_axis.shape[0]))

        last_profiles = None

        for i, phi in enumerate(potential):
            sol = self.odesolve_dirichlet(x_axis, y_initial, phi, tol=tol)
            last_profiles = self.compute_profiles(sol)
            chg[i] = last_profiles.efield[0] * C.EPS_0 * last_profiles.eps[0]
            efield[i] = last_profiles.efield[0]
            c_surf['Cations'][i] = last_profiles.c_dict['Cations'][0]
            c_surf['Anions'][i] = last_profiles.c_dict['Anions'][0]
            c_surf['Solvent'][i] = last_profiles.c_dict['Solvent'][0]
            max_res[i] = np.max(sol.rms_residuals)

            x_axis = sol.x
            y_initial = sol.y

        print(f"Sweep from {potential[0]:.2f}V to {potential[-1]:.2f}V. " \
            + f"Maximum relative residual: {np.max(max_res):.5e}.")
        return chg, efield, c_surf, last_profiles

    def sweep(self, potential: np.ndarray, tol: float=1e-3):
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """
        # Find potential closest to PZC
        i_pzc = np.argmin(np.abs(potential)).squeeze()

        c_surf = {'Cations': np.zeros(potential.shape),
                  'Anions': np.zeros(potential.shape),
                  'Solvent': np.zeros(potential.shape)}
        chg = np.zeros(potential.shape)
        efield = np.zeros(potential.shape)

        chg_neg, efield_neg, c_surf_neg, _ = self.sequential_solve(potential[i_pzc::-1], tol)
        chg[:i_pzc+1] = chg_neg[::-1]
        efield[:i_pzc+1] = efield_neg[::-1]
        c_surf['Cations'][:i_pzc+1] = c_surf_neg['Cations'][::-1]
        c_surf['Anions'][:i_pzc+1] = c_surf_neg['Anions'][::-1]
        c_surf['Solvent'][:i_pzc+1] = c_surf_neg['Solvent'][::-1]

        chg[i_pzc:], efield[i_pzc:], c_surf_pos, _ = self.sequential_solve(potential[i_pzc::1], tol)
        c_surf['Cations'][i_pzc:] = c_surf_pos['Cations']
        c_surf['Anions'][i_pzc:] = c_surf_pos['Anions']
        c_surf['Solvent'][i_pzc:] = c_surf_pos['Solvent']

        cap = np.gradient(chg, edge_order=2)/np.gradient(potential) * 1e2
        return PotentialSweepSolution(phi=potential, charge=chg, cap=cap, c_surf=c_surf, efield=efield, name=self.name)

    def spatial_profiles(self, phi0: float, tol: float=1e-3):
        """
        Get spatial profiles solution struct.
        """
        sign = phi0/abs(phi0)
        _, _, _, profiles = self.sequential_solve(np.arange(0, phi0, sign*0.01), tol)
        return profiles

    @abstractmethod
    def compute_profiles(self, sol) -> SpatialProfilesSolution:
        """
        Convert a dimensionless scipy solution into dimensional spatial
        profiles
        """


class GouyChapman(DoubleLayerModel):
    """
    Gouy-Chapman model, treating ions as point particles obeying Boltzmann statistics.
    See for example Schmickler & Santos' Interfacial Electrochemistry.
    """
    def __init__(self, ion_concentration_molar: float) -> None:
        super().__init__(ion_concentration_molar)
        self.name = f'Gouy-Chapman {self.c_0:.3f}M'

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        dy2 = np.sinh(y[0, :])
        return np.vstack([dy1, dy2])

    def compute_profiles(self, sol):
        ret = SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                'Cations': self.c_0 * np.exp(-sol.y[0, :]),
                'Anions': self.c_0 * np.exp(sol.y[0, :]),
                'Solvent': np.zeros(sol.x.shape)},
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret

    def analytical_sweep(self, potential: np.ndarray):
        """
        Analytic solution to a potential sweep in the Gouy-Chapman model.

        ion_concentration_molar: bulk ion concentration in molar
        potential: potential array in V
        """
        c_surf = {'Cations': np.zeros(potential.shape),
                  'Anions': np.zeros(potential.shape),
                  'Solvent': np.zeros(potential.shape)}

        cap = self.kappa_debye * C.EPS_R_WATER * C.EPS_0 * \
            np.cosh(C.BETA*C.Z*C.E_0 * potential / 2) * 1e2
        chg = np.sqrt(8 * self.n_0 * C.K_B * C.T * C.EPS_R_WATER * C.EPS_0) * \
            np.sinh(C.BETA * C.Z * C.E_0 * potential / 2)
        efield = np.zeros(potential.shape)
        return PotentialSweepSolution(
            phi=potential, charge=chg, cap=cap, c_surf=c_surf, efield=efield, name=self.name + ' (Analytic)')


class Borukhov(DoubleLayerModel):
    """
    Model developed by Borukhov, Andelman and Orland, modifying the Guy-Chapman model to
    take finite ion size into account.
    https://doi.org/10.1016/S0013-4686(00)00576-4
    """
    def __init__(self, ion_concentration_molar: float, a_m: float) -> None:
        super().__init__(ion_concentration_molar)
        self.chi_0 = 2 * a_m ** 3 * self.n_0
        self.n_max = 1/a_m**3
        self.name = f'Borukhov {self.c_0:.3f}M {a_m*1e10:.1f}Å'

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        dy2 = np.sinh(y[0, :]) / (1 - self.chi_0 + self.chi_0 * np.cosh(y[0, :]))
        return np.vstack([dy1, dy2])

    def dirichlet(self, phi0):
        return lambda ya, yb: np.array([ya[0] - C.BETA * C.Z * C.E_0 * phi0, yb[0]])

    def get_dimensionless_x_axis(self, x_axis_nm):
        """
        Return dimensionless x-axis using model length scale.
        """
        return self.kappa_debye * 1e-9 * x_axis_nm

    def compute_profiles(self, sol):
        bf_c = np.exp(-sol.y[0, :])
        bf_a = np.exp(sol.y[0, :])
        denom = 1 - self.chi_0 + self.chi_0 * np.cosh(sol.y[0, :])

        # Return solution struct
        ret = SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                'Cations': self.c_0 * bf_c / denom,
                'Anions': self.c_0 * bf_a / denom,
                'Solvent': np.zeros(sol.x.shape)
            },
            eps=np.ones(sol.x.shape) * C.EPS_R_WATER,
            name=self.name)
        return ret

    def analytical_sweep(self, potential: np.ndarray):
        """
        Analytic solution to a potential sweep in the Borukhov-Andelman-Orland model.

        ion_concentration_molar: bulk ion concentration in molar
        a_m: ion diameter in m
        potential: potential array in V
        """
        c_surf = {'Cations': np.zeros(potential.shape),
                  'Anions': np.zeros(potential.shape),
                  'Solvent': np.zeros(potential.shape)}

        y_0 = C.BETA*C.Z*C.E_0*potential  # dimensionless potential
        chg = np.sqrt(4 * C.K_B * C.T * C.EPS_0 * C.EPS_R_WATER * self.n_0 / self.chi_0) \
            * np.sqrt(np.log(self.chi_0 * np.cosh(y_0) - self.chi_0 + 1)) * y_0 / np.abs(y_0)
        cap = np.sqrt(2) * self.kappa_debye * C.EPS_R_WATER * C.EPS_0 / np.sqrt(self.chi_0) \
            * self.chi_0 * np.sinh(np.abs(y_0)) \
            / (self.chi_0 * np.cosh(y_0) - self.chi_0 + 1) \
            / (2*np.sqrt(np.log(self.chi_0 * np.cosh(y_0) - self.chi_0 + 1))) \
            * 1e2 # uF/cm^2
        efield = np.zeros(potential.shape)

        return PotentialSweepSolution(
            phi=potential, charge=chg, cap=cap, c_surf=c_surf, efield=efield, name=self.name + ' (Analytic)')


class Abrashkin(DoubleLayerModel):
    """
    Langevin-Poisson-Boltzmann model that was derived by Abrashkin, further developed by
    Gongadze & Iglic (here implemented without refractive index-dependent factors), and
    related to kinetics by Jun Huang.
    https://doi.org/10.1103/PhysRevLett.99.077801
    https://doi.org/10.1016/j.electacta.2015.07.179
    https://doi.org/10.1021/jacsau.1c00315
    """
    def __init__(self,
                 ion_concentration_molar: float,
                 gamma_c: float, gamma_a: float,
                 eps_r_opt=C.N_WATER**2) -> None:
        super().__init__(ion_concentration_molar)
        self.gamma_c = gamma_c
        self.gamma_a = gamma_a
        self.n_max = C.C_WATER_BULK * 1e3 * C.N_A
        n_s_0 = self.n_max - gamma_c * self.n_0 - gamma_a * self.n_0

        self.chi = self.n_0 / self.n_max
        self.chi_s = n_s_0 / self.n_max

        self.eps_r_opt = eps_r_opt

        self.p_water = np.sqrt(3 * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / (C.BETA * self.n_max))
        self.p_tilde = self.p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = f'LPB {self.c_0:.3f}M {gamma_c:.1f}-{gamma_a:.1f}'

    def densities(self, sol_y):
        """
        Compute cation, anion and solvent densities.
        """
        bf_c = np.exp(-sol_y[0, :])
        bf_a = np.exp(+sol_y[0, :])
        bf_s = L.sinh_x_over_x(self.p_tilde * sol_y[1, :])
        denom = self.chi_s * bf_s + self.gamma_c * self.chi * bf_c + self.gamma_a * self.chi * bf_a
        n_cat = self.n_0 * bf_c / denom
        n_an  = self.n_0 * bf_a / denom
        n_sol = self.n_max * self.chi_s * bf_s / denom
        return n_cat, n_an, n_sol

    def ode_rhs(self, x, y):
        dy1 = y[1, :]
        n_cat, n_an, n_sol = self.densities(y)

        numer1 = n_an - n_cat
        numer2 = self.p_tilde * y[1, :] * L.langevin_x(self.p_tilde * y[1, :]) * \
            (self.gamma_a * n_an - self.gamma_c * n_cat) * n_sol/self.n_max
        denom1 = self.kappa_debye ** 2 * self.eps_r_opt * C.EPS_0 / (C.Z * C.E_0)**2 / C.BETA
        denom2 = self.p_tilde**2 * n_sol * L.d_langevin_x(self.p_tilde * y[1, :])
        denom3 = self.p_tilde**2 * L.langevin_x(self.p_tilde * y[1, :])**2 * \
            (self.gamma_c * n_cat + self.gamma_a * n_an) * n_sol/self.n_max

        dy2 = (numer1 + numer2) / (denom1 + denom2 + denom3)
        return np.vstack([dy1, dy2])

    def permittivity(self, sol_y):
        """
        Compute the permittivity using the electric field
        n_sol: solvent number density
        y_1: dimensionless electric field
        """
        sol_y = np.atleast_1d(sol_y).reshape(2, -1)
        _, _, n_sol = self.densities(sol_y)
        two_nref_over_epsrw = self.kappa_debye ** 2 * C.EPS_0 / (C.Z * C.E_0)**2 / C.BETA
        return self.eps_r_opt + \
               self.p_tilde**2 * n_sol / two_nref_over_epsrw * \
               L.langevin_x_over_x(self.p_tilde * sol_y[1, :])

    def compute_profiles(self, sol):
        n_cat, n_an, n_sol = self.densities(sol.y)

        # Return solution struct
        ret = SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                'Cations': n_cat/1e3/C.N_A,
                'Anions': n_an/1e3/C.N_A,
                'Solvent': n_sol/1e3/C.N_A
            },
            eps=self.permittivity(sol.y),
            name=self.name)

        return ret


class ProtonLPB(DoubleLayerModel):
    """
    Taking into account protons and hydroxy ions
    """
    def __init__(self,
                 support_ion_concentration_molar: float,
                 gamma_c: float, gamma_a: float,
                 gamma_h: float, gamma_oh: float,
                 eps_r_opt=C.N_WATER**2) -> None:
        self.gammas = np.array([gamma_h, gamma_oh, gamma_c, gamma_a, 1]).reshape(5, 1)
        self.charge = np.array([+1, -1, +1, -1, 0]).reshape(5, 1)

        # Nondimensional quantities are based on debye length with support ion concentration
        super().__init__(support_ion_concentration_molar)
        self.n_max = C.C_WATER_BULK * 1e3 * C.N_A
        self.eps_r_opt = eps_r_opt
        self.kappa_debye = np.sqrt(2*self.n_max*(C.Z*C.E_0)**2 /
                                   (C.EPS_0*C.EPS_R_WATER*C.K_B*C.T))

        p_water = np.sqrt(3 * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / (C.BETA * self.n_max))
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = f'pH-LPB {self.c_0:.3f}M {gamma_c:.1f}-{gamma_a:.1f}'

    def densities(self, sol_y: np.ndarray, p_h: float):
        """
        Compute cation, anion and solvent densities.
        """
        # Compute bulk number densities
        c_bulk = np.zeros((5, 1))
        c_bulk[0] = 10 ** (- p_h)           # [H+]
        c_bulk[1] = 10 ** (- C.PKW + p_h)   # [OH-]
        c_bulk[2] = self.c_0 + c_bulk[1]    # [Cat]
        c_bulk[3] = self.c_0 + c_bulk[0]    # [An]
        c_bulk[4] = C.C_WATER_BULK - np.sum(self.gammas * c_bulk) # [H2O]
        n_bulk = c_bulk * 1e3 * C.N_A

        # Compute chi for each species
        chi = n_bulk / self.n_max

        # Initialize array for profiles
        n_profile = np.zeros((5, sol_y.shape[1]))

        # Asymptotic case: large negative electrode potential
        big_neg = sol_y[0, :] < -1
        bf_combined = L.sinh_x1_over_x1_times_exp_x2(self.p_tilde*sol_y[1,big_neg], sol_y[0,big_neg])
        denom_combined = chi[4] * bf_combined + self.gammas[0]*chi[0] + self.gammas[2]*chi[2] \
            + self.gammas[1]*chi[1]*np.exp(+2*sol_y[0, big_neg]) \
            + self.gammas[3]*chi[3]*np.exp(+2*sol_y[0, big_neg])
        n_profile[0, big_neg] = n_bulk[0] / denom_combined
        n_profile[1, big_neg] = n_bulk[1]*np.exp(+2*sol_y[0, big_neg]) / denom_combined
        n_profile[2, big_neg] = n_bulk[2] / denom_combined
        n_profile[3, big_neg] = n_bulk[3]*np.exp(+2*sol_y[0, big_neg]) / denom_combined
        n_profile[4, big_neg] = n_bulk[4] * bf_combined / denom_combined

        # Asymptotic case: large positive electrode potential and electric field
        big_pos = sol_y[0, :] > 1
        bf_combined = L.sinh_x1_over_x1_times_exp_x2(self.p_tilde*sol_y[1,big_pos], -sol_y[0,big_pos])
        denom_combined = chi[4] * bf_combined + self.gammas[1]*chi[1] + self.gammas[3]*chi[3] \
            + self.gammas[0]*chi[0]*np.exp(-2*sol_y[0, big_pos]) \
            + self.gammas[2]*chi[2]*np.exp(-2*sol_y[0, big_pos])
        n_profile[0, big_pos] = n_bulk[0]*np.exp(-2*sol_y[0, big_pos]) / denom_combined
        n_profile[1, big_pos] = n_bulk[1] / denom_combined
        n_profile[2, big_pos] = n_bulk[2]*np.exp(-2*sol_y[0, big_pos]) / denom_combined
        n_profile[3, big_pos] = n_bulk[3] / denom_combined
        n_profile[4, big_pos] = n_bulk[4] * bf_combined / denom_combined

        # General case: compute Boltzmann factors
        bf_pos = np.exp(-sol_y[0, ~big_neg * ~big_pos])
        bf_neg = np.exp(+sol_y[0, ~big_neg * ~big_pos])
        bf_sol = L.sinh_x_over_x(self.p_tilde * sol_y[1, ~big_neg * ~big_pos])
        bfs = np.array([bf_pos, bf_neg, bf_pos, bf_neg, bf_sol]) # shape (5, ...)

        # Compute denominator
        denom = np.sum(self.gammas * chi * bfs, axis=0)

        # Compute profiles
        n_profile[:, ~big_neg * ~big_pos] = n_bulk * bfs / denom

        return n_profile

    def get_lambda_ode_rhs(self, p_h):
        """
        Get the ODE RHS as a lambda function to pass to Scipy's solve_bvp, given a pH
        """
        return lambda x, y: self.ode_rhs(x, y, p_h)

    def ode_rhs(self, x, y, p_h: float=7):
        dy1 = y[1, :]
        n_arr = self.densities(y, p_h)

        numer1 = np.sum(-self.charge * n_arr, axis=0)
        numer2 = self.p_tilde * y[1, :] * L.langevin_x(self.p_tilde * y[1, :]) * \
            np.sum(n_arr * -self.charge * self.gammas, axis=0) * n_arr[4]/self.n_max
        denom1 = self.kappa_debye ** 2 * self.eps_r_opt * C.EPS_0 / (C.Z * C.E_0)**2 / C.BETA
        denom2 = self.p_tilde**2 * n_arr[4] * L.d_langevin_x(self.p_tilde * y[1, :])
        denom3 = self.p_tilde**2 * L.langevin_x(self.p_tilde * y[1, :])**2 * \
            np.sum(n_arr * self.charge**2 * self.gammas, axis=0) * n_arr[4]/self.n_max

        dy2 = (numer1 + numer2) / (denom1 + denom2 + denom3)
        return np.vstack([dy1, dy2])

    def permittivity(self, sol_y: np.ndarray, n_sol: np.ndarray):
        """
        Compute the permittivity using the electric field
        n_sol: solvent number density
        y_1: dimensionless electric field
        """
        two_nref_over_epsrw = self.kappa_debye ** 2 * C.EPS_0 / (C.Z * C.E_0)**2 / C.BETA
        return self.eps_r_opt + \
               self.p_tilde**2 * n_sol / two_nref_over_epsrw * \
               L.langevin_x_over_x(self.p_tilde * sol_y[1, :])

    def compute_profiles(self, sol, p_h: float=7):
        n_arr = self.densities(sol.y, p_h)

        # Return solution struct
        ret = SpatialProfilesSolution(
            x=sol.x / self.kappa_debye * 1e9,
            phi=sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            efield=-sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            c_dict={
                r'H$^+$': n_arr[0]/1e3/C.N_A,
                r'OH$^-$': n_arr[1]/1e3/C.N_A,
                'Cations': n_arr[2]/1e3/C.N_A,
                'Anions': n_arr[3]/1e3/C.N_A,
                'Solvent': n_arr[4]/1e3/C.N_A
            },
            eps=self.permittivity(sol.y, n_arr[4]),
            name=self.name)

        return ret

    def get_insulator_bc_lambda(self, p_h):
        """
        Return a boundary condition function to pass to scipy's solve_bvp
        """
        return lambda ya, yb: self.insulator_bc(ya, yb, p_h)

    def insulator_bc(self, ya, yb, p_h):
        """
        Boundary condition
        """
        #pylint: disable=invalid-name
        n_arr = self.densities(ya.reshape(2, 1), p_h)
        c_arr = n_arr / 1e3 / C.N_A
        eps_r = self.permittivity(ya.reshape(2, 1), np.atleast_1d(n_arr[4]))

        left = 2 * eps_r * ya[1] \
            + self.kappa_debye * C.EPS_R_WATER * C.N_SITES_SILICA / self.n_max \
            * (c_arr[0]**2 - C.K_SILICA_A * C.K_SILICA_B) \
            / (C.K_SILICA_A * C.K_SILICA_B + C.K_SILICA_B * c_arr[0] + c_arr[0]**2)
        right = yb[0]

        return np.array([left.squeeze(), right])

    def sequential_solve_ins(self, ph_range: np.ndarray, tol: float=1e-3):
        """
        Sweep over a potential array and use the previous solution as initial
        condition for the next.

        Returns: charge for each potential; last solution
        """
        chg = np.zeros(ph_range.shape)
        phi = np.zeros(ph_range.shape)
        c_h = np.zeros(ph_range.shape)
        max_res = np.zeros(ph_range.shape)

        x_axis = self.create_x_mesh(10, 1000)
        y_initial = np.zeros((2, x_axis.shape[0]))

        last_profiles = None

        for i, p_h in enumerate(ph_range):
            sol = solve_bvp(
                self.get_lambda_ode_rhs(p_h),
                self.get_insulator_bc_lambda(p_h),
                x_axis,
                y_initial,
                tol=tol,
                max_nodes=int(1e8),
                verbose=0)
            last_profiles = self.compute_profiles(sol, p_h)
            chg[i] = last_profiles.efield[0] * C.EPS_0 * last_profiles.eps[0]
            phi[i] = last_profiles.phi[0]
            c_h[i] = last_profiles.c_dict[r'H$^+$'][0]
            max_res[i] = np.max(sol.rms_residuals)

            x_axis = sol.x
            y_initial = sol.y

        print(f"Sweep from pH {ph_range[0]:.2f} to {ph_range[-1]:.2f}. " \
            + f"Maximum relative residual: {np.max(max_res):.5e}.")
        return phi, chg, c_h, last_profiles

    def sweep_ins(self, ph_range: np.ndarray, tol: float=1e-3):
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """
        # Find pH closest to PZC
        ph_pzc = -1/2 * np.log10(C.K_SILICA_A*C.K_SILICA_B)
        i_pzc = np.argmin(np.abs(ph_range - ph_pzc)).squeeze()

        chg = np.zeros(ph_range.shape)
        phi = np.zeros(ph_range.shape)
        c_h = np.zeros(ph_range.shape)

        phi_neg, chg_neg, c_h_neg, _ = self.sequential_solve_ins(ph_range[i_pzc::-1], tol)
        chg[:i_pzc+1] = chg_neg[::-1]
        phi[:i_pzc+1] = phi_neg[::-1]
        c_h[:i_pzc+1] = c_h_neg[::-1]
        phi[i_pzc:], chg[i_pzc:], c_h[i_pzc:], _ = self.sequential_solve_ins(ph_range[i_pzc::1],
                                                                             tol)
        cap = np.gradient(chg, edge_order=2)/np.gradient(phi) * 1e2
        return PhSweepSolution(phi=phi, charge=chg, cap=cap, c_h=c_h, name=self.name)

    def spatial_profiles_ins(self, p_h: float, tol: float=1e-3):
        """
        Get spatial profiles solution struct.
        """
        ph_pzc = -1/2 * np.log10(C.K_SILICA_A*C.K_SILICA_B)
        sign = (p_h - ph_pzc)/abs(p_h - ph_pzc)
        _, _, _, profiles = self.sequential_solve_ins(np.arange(ph_pzc, p_h, sign*0.01), tol)
        return profiles

    def odesolve_dirichlet(self,
            x_axis: np.ndarray,
            y_initial: np.ndarray,
            phi0: float,
            tol: float=1e-3,
            p_h: float=7):
        """
        Wrapper for scipy's solve_bvp, using the class methods
        """
        sol = solve_bvp(
            self.get_lambda_ode_rhs(p_h),
            self.dirichlet(phi0),
            x_axis,
            y_initial,
            tol=tol,
            max_nodes=int(1e8),
            verbose=0)
        return sol

    def sequential_solve(self, potential: np.ndarray, tol: float=1e-3, p_h: float=7):
        """
        Sweep over a potential array and use the previous solution as initial
        condition for the next.

        Returns: charge for each potential; last solution
        """
        chg = np.zeros(potential.shape)
        max_res = np.zeros(potential.shape)

        x_axis = self.create_x_mesh(100, 1000)
        y_initial = np.zeros((2, x_axis.shape[0]))

        last_profiles = None

        for i, phi in enumerate(potential):
            sol = self.odesolve_dirichlet(x_axis, y_initial, phi, tol=tol, p_h=p_h)
            last_profiles = self.compute_profiles(sol, p_h=p_h)
            chg[i] = last_profiles.efield[0] * C.EPS_0 * last_profiles.eps[0]
            max_res[i] = np.max(sol.rms_residuals)

            x_axis = sol.x
            y_initial = sol.y

        print(f"Sweep from {potential[0]:.2f}V to {potential[-1]:.2f}V. " \
            + f"Maximum relative residual: {np.max(max_res):.5e}.")
        return chg, last_profiles

    def sweep(self, potential: np.ndarray, tol: float=1e-3, p_h: float=7):
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """
        # Find potential closest to PZC
        i_pzc = np.argmin(np.abs(potential)).squeeze()

        chg = np.zeros(potential.shape)
        chg_neg, _ = self.sequential_solve(potential[i_pzc::-1], tol, p_h)
        chg[:i_pzc+1] = chg_neg[::-1]
        chg[i_pzc:], _ = self.sequential_solve(potential[i_pzc::1], tol, p_h)

        cap = np.gradient(chg, edge_order=2)/np.gradient(potential) * 1e2

        c_surf = {'Cations': np.zeros(potential.shape),
                  'Anions': np.zeros(potential.shape),
                  'Solvent': np.zeros(potential.shape)}
        efield = np.zeros(potential.shape)

        return PotentialSweepSolution(phi=potential, charge=chg, cap=cap, c_surf=c_surf, efield=efield, name=self.name)

    def spatial_profiles(self, phi0: float, tol: float=1e-3, p_h: float=7):
        """
        Get spatial profiles solution struct.
        """
        sign = phi0/abs(phi0)
        _, profiles = self.sequential_solve(np.arange(0, phi0, sign*0.01), tol, p_h)
        return profiles