"""
Implementation of double-layer models
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, cumulative_trapezoid

import constants as C
import langevin as L

class Aqueous:
    """
    Taking into account protons and hydroxy ions
    """
    def __init__(self,
                 support_ion_concentration_molar: float,
                 gamma_c: float, gamma_a: float,
                 gamma_h: float, gamma_oh: float,
                 eps_r_opt=C.N_WATER**2) -> None:
        self.c_0 = support_ion_concentration_molar
        self.n_0 = self.c_0 * 1e3 * C.N_A
        self.n_max = C.C_WATER_BULK * 1e3 * C.N_A

        self.gammas = np.array([gamma_h, gamma_oh, gamma_c, gamma_a, 1]).reshape(5, 1)
        self.charge = np.array([+1, -1, +1, -1, 0]).reshape(5, 1)

        self.eps_r_opt = eps_r_opt
        self.kappa_debye = np.sqrt(2*self.n_max*(C.Z*C.E_0)**2 /
                                   (C.EPS_0*C.EPS_R_WATER*C.K_B*C.T))

        p_water = np.sqrt(3 * (C.EPS_R_WATER - self.eps_r_opt) * C.EPS_0 / (C.BETA * self.n_max))
        self.p_tilde = p_water * self.kappa_debye / (C.Z * C.E_0)

        self.name = f'EDL {self.c_0:.3f}M {gamma_c:.0f}-{gamma_a:.0f}'

    def create_x_mesh(self, xmax_nm, n_points):
        """
        Get a logarithmically spaced x-axis, fine mesh close to electrode
        """
        max_exponent = np.log10(xmax_nm)
        x_nm = np.logspace(-6, max_exponent, n_points) - 1e-6
        return self.kappa_debye * 1e-9 * x_nm

    def ode_lambda(self, p_h):
        """
        Get an ODE RHS function to pass to Scipy's solve_bvp; using lambda gives
        the possibility to include specific parameters like pH in derived classes
        """
        return lambda x, y: self.ode_rhs(x, y, p_h)

    def potential_sequential_solve(self,
                         ode_lambda: callable,
                         potential: np.ndarray,
                         tol: float=1e-3,
                         p_h: float=7) -> list[pd.DataFrame]:
        """
        Sweep over a Dirichlet boundary condition for the potential and use the previous
        solution as initial condition for the next. The initial condition assumes that we
        start at the PZC.

        Returns: charge for each parameter; list of SpatialProfilesSolutions
        """
        df_list = []
        max_res = np.zeros(potential.shape)

        x_axis = self.create_x_mesh(100, 1000)
        y_initial = np.zeros((2, x_axis.shape[0]))

        for i, phi in enumerate(potential):
            sol = solve_bvp(
                ode_lambda(p_h),
                self.dirichlet_lambda(phi),
                x_axis,
                y_initial,
                tol=tol,
                max_nodes=int(1e8),
                verbose=0)
            profile_df = self.compute_profiles(sol, p_h)
            max_res[i] = np.max(sol.rms_residuals)
            df_list.append(profile_df)

            x_axis = sol.x
            y_initial = sol.y

        print(f"Sweep from {potential[0]:.2f}V to {potential[-1]:.2f}V. " \
            + f"Maximum relative residual: {np.max(max_res):.5e}.")
        return df_list

    def potential_sweep(self, potential: np.ndarray, tol: float=1e-3, p_h: float=7) -> pd.DataFrame:
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """
        # Find potential closest to PZC
        i_pzc = np.argmin(np.abs(potential)).squeeze()

        profiles_neg = self.potential_sequential_solve(
            self.ode_lambda,
            potential[i_pzc::-1],
            tol=tol,
            p_h=p_h)
        profiles_pos = self.potential_sequential_solve(
            self.ode_lambda,
            potential[i_pzc::1],
            tol)

        all_profiles = profiles_neg[::-1] + profiles_pos[1:]

        # Create dataframe with interface values (iloc[0]) to return
        sweep_df = pd.concat(
            [pd.DataFrame(profile.iloc[0]).transpose() for profile in all_profiles],
            ignore_index=True)
        sweep_df.index.name = self.name

        sweep_df['phi0'] = potential
        sweep_df['charge'] = sweep_df['efield'] * C.EPS_0 * sweep_df['eps']
        sweep_df['capacity'] = np.gradient(sweep_df['charge'], edge_order=2) \
            / np.gradient(potential)
        if 'phi' in sweep_df.columns:
            half_ctrion_thickness = 1/2 * (self.gammas[2,0] / self.n_max) ** (1/3)
            phi_rp = sweep_df['phi'] + sweep_df['efield'] * (half_ctrion_thickness - C.D_ADSORBATE_LAYER)
            sweep_df['phi2'] = phi_rp
        if 'x' in sweep_df.columns:
            sweep_df.drop('x', inplace=True, axis=1)

        return sweep_df

    def spatial_profiles(self, phi0: float, p_h: float=7, tol: float=1e-3) -> pd.DataFrame:
        """
        Get spatial profiles solution struct.
        """
        sign = phi0/abs(phi0)
        profiles = self.potential_sequential_solve(self.ode_lambda,
                                                      np.arange(0, phi0, sign*0.01),
                                                      tol=tol,
                                                      p_h=p_h)
        return profiles[-1]

    def get_stern_layer_thickness(self, phi0):
        """
        Calculate the thickness of the Stern layer as half of the effective ion size
        """
        half_ctrion_thickness = None
        if phi0 < 0:
            half_ctrion_thickness = 1/2 * (self.gammas[2,0] / self.n_max) ** (1/3)
        else:
            half_ctrion_thickness = 1/2 * (self.gammas[3,0] / self.n_max) ** (1/3)
        return half_ctrion_thickness

    def dirichlet_lambda(self, phi0):
        """
        Return a boundary condition function to pass to scipy's solve_bvp
        """
        # old: return lambda ya, yb: np.array([ya[0] - C.BETA * C.Z * C.E_0 * phi0, yb[0]])
        return lambda ya, yb: np.array(
            [ya[0] - C.BETA * C.Z * C.E_0 * phi0 \
             - ya[1] * self.kappa_debye * self.get_stern_layer_thickness(phi0),
             yb[0]
            ])


    def bulk_densities(self, p_h: float):
        """
        Compute bulk cation, anion and solvent densities.
        """
        # Compute bulk number densities
        c_bulk = np.zeros((5, 1))
        c_bulk[0] = 10 ** (- p_h)                         # [H+]
        c_bulk[1] = 10 ** (- C.PKW + p_h)                 # [OH-]
        c_bulk[2] = self.c_0 - c_bulk[0]                  # [Cat]
        c_bulk[3] = c_bulk[0] + c_bulk[2] - c_bulk[1]     # [An]
        c_bulk[4] = C.C_WATER_BULK - np.sum(self.gammas * c_bulk) # [H2O]
        n_bulk = c_bulk * 1e3 * C.N_A
        return n_bulk

    def boltzmann_factors(self, sol_y):
        """
        Compute cation, anion and solvent Boltzmann factors.
        """
        bf_pos = np.exp(-sol_y[0, :])
        bf_neg = np.exp(+sol_y[0, :])
        bf_sol = L.sinh_x_over_x(self.p_tilde * sol_y[1, :])
        bfs = np.array([bf_pos, bf_neg, bf_pos, bf_neg, bf_sol]) # shape (5, ...)
        return bfs

    def densities(self, sol_y: np.ndarray, p_h: float):
        """
        Compute cation, anion and solvent densities.
        """
        n_profile = self.bulk_densities(p_h) \
            * self.boltzmann_factors(sol_y) / self.get_denominator(sol_y, p_h)

        return n_profile

    def get_denominator(self, sol_y: np.ndarray, p_h: float):
        """
        Get denominator of densities
        """
        n_bulk = self.bulk_densities(p_h)
        chi = n_bulk / self.n_max

        bfs = self.boltzmann_factors(sol_y)
        denom = np.sum(self.gammas * chi * bfs, axis=0)

        return denom

    def ode_rhs(self, x, y, p_h): #pylint: disable=unused-argument
        """
        ode rhs to be solved by solve_bvp
        """
        dy1 = y[1, :]
        n_arr = self.densities(y, p_h)

        numer1 = np.sum(-self.charge * n_arr, axis=0)
        numer2 = self.p_tilde * y[1, :] * L.langevin_x(self.p_tilde * y[1, :]) * \
            np.sum(n_arr * -self.charge * self.gammas, axis=0) * n_arr[4]/self.n_max
        denom1 = 2*self.n_max*self.eps_r_opt/C.EPS_R_WATER
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
        return self.eps_r_opt + \
               C.EPS_R_WATER * self.p_tilde**2 * n_sol / (2*self.n_max) * \
               L.langevin_x_over_x(self.p_tilde * sol_y[1, :])

    def compute_profiles(self, sol, p_h: float) -> pd.DataFrame:
        """
        compute spatial profiles
        """
        n_arr = self.densities(sol.y, p_h)

        result = pd.DataFrame({
            'x':  sol.x / self.kappa_debye * 1e9,
            'phi':  sol.y[0, :] / (C.BETA * C.Z * C.E_0),
            'efield': -sol.y[1, :] * self.kappa_debye / (C.BETA * C.Z * C.E_0),
            'protons': n_arr[0]/1e3/C.N_A,
            'hydroxide': n_arr[1]/1e3/C.N_A,
            'cations': n_arr[2]/1e3/C.N_A,
            'anions': n_arr[3]/1e3/C.N_A,
            'solvent': n_arr[4]/1e3/C.N_A,
            'eps': self.permittivity(sol.y, n_arr[4]),
            'charge density': C.E_0 * np.sum(n_arr * self.charge, axis=0)})
        result['pressure'] = cumulative_trapezoid(result['charge density'][::-1] * result['efield'][::-1], x=result['x'][::-1]*1e-9, initial=0)[::-1]
        result.index.name = self.name

        return result

    def insulator_bc_lambda(self, p_h):
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
        eps_r = self.permittivity(ya.reshape(2, 1), np.atleast_1d(n_arr[4]))

        c_bulk_arr = self.bulk_densities(p_h) / 1e3 / C.N_A

        left = 2 * eps_r * ya[1] \
            - self.kappa_debye * C.EPS_R_WATER * C.N_SITES_SILICA / self.n_max \
            * C.K_SILICA_A / (C.K_SILICA_A + c_bulk_arr[0] * \
                              np.exp(-ya[0] + ya[1] * self.kappa_debye * \
                                     self.get_stern_layer_thickness(ya[0])))

        right = yb[0]

        return np.array([left.squeeze(), right])

    def insulator_sequential_solve(self,
                         ph_range: np.ndarray,
                         tol: float=1e-3) -> list[pd.DataFrame]:
        """
        Sweep over a boundary condition parameter array (potential, pH) and use the previous
        solution as initial condition for the next. The initial condition assumes that we
        start at the PZC.

        Returns: charge for each parameter; list of SpatialProfilesSolutions
        """
        profile_list = []
        max_res = np.zeros(ph_range.shape)

        x_axis = self.create_x_mesh(100, 1000)
        y_initial = np.zeros((2, x_axis.shape[0]))

        for i, p_h in enumerate(ph_range):
            sol = solve_bvp(
                self.ode_lambda(p_h),
                self.insulator_bc_lambda(p_h),
                x_axis,
                y_initial,
                tol=tol,
                max_nodes=int(1e8),
                verbose=0)
            prf = self.compute_profiles(sol, p_h)
            max_res[i] = np.max(sol.rms_residuals)
            profile_list.append(prf)

            x_axis = sol.x
            y_initial = sol.y

        print(f"Sweep from pH {ph_range[0]:.2f} to {ph_range[-1]:.2f}. " \
            + f"Maximum relative residual: {np.max(max_res):.5e}.")
        return profile_list

    def ph_sweep(self, ph_range: np.ndarray, tol: float=1e-3) -> pd.DataFrame:
        """
        Numerical solution to a potential sweep for a defined double-layer model.
        """
        # Find pH closest to PZC
        ph_pzc = -1/2 * np.log10(C.K_SILICA_A*C.K_SILICA_B)
        i_pzc = np.argmin(np.abs(ph_range - ph_pzc)).squeeze()

        profiles_neg = self.insulator_sequential_solve(
            ph_range[i_pzc::-1],
            tol)
        profiles_pos = self.insulator_sequential_solve(
            ph_range[i_pzc::1],
            tol)

        all_profiles = profiles_neg[::-1] + profiles_pos[1:]

        # Create dataframe with interface values (iloc[0]) to return
        sweep_df = pd.concat(
            [pd.DataFrame(profile.iloc[0]).transpose() for profile in all_profiles],
            ignore_index=True)
        sweep_df.index.name = self.name

        sweep_df['ph'] = ph_range
        sweep_df['charge'] = sweep_df['efield'] * C.EPS_0 * sweep_df['eps']
        if 'phi' in sweep_df.columns:
            sweep_df.rename({'phi': 'phi2'}, inplace=True, axis=1)
            sweep_df['phi0'] = sweep_df['phi2'] + sweep_df['efield'] * C.D_ADSORBATE_LAYER
        if 'x' in sweep_df.columns:
            sweep_df.drop('x', inplace=True, axis=1)

        return sweep_df

    def insulator_spatial_profiles(self, p_h: float, tol: float=1e-3) -> pd.DataFrame:
        """
        Get spatial profiles solution struct.
        """
        ph_pzc = -1/2 * np.log10(C.K_SILICA_A*C.K_SILICA_B)
        sign = (p_h - ph_pzc)/abs(p_h - ph_pzc)
        profiles = self.insulator_sequential_solve(
            np.arange(ph_pzc, p_h, sign*0.1),
            tol)
        return profiles[-1]
