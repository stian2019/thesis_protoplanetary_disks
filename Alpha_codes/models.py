import datetime
import math

import astropy.constants as const
import astropy.units as units
import numpy as np
from scipy import interpolate

try:
    from numba import njit, jit
except ImportError:
    warnings.warn('numba not available, calculation will be ~8x slower; better install numba!')


    def njit(ob):
        return ob

'''
all constants are in units of cgs
'''
GM_sun = const.GM_sun.cgs.value
M_sun = const.M_sun.cgs.value
Year = units.year.to(units.s)
Au = const.au.cgs.value
G = const.G.cgs.value
StefanBC = const.sigma_sb.cgs.value
R = const.R.cgs.value
M_p = const.m_p.cgs.value
mu = 2.3
FLOOR = 1e-100

'''
general functions
'''


def derivative(values, r_here):
    return (values[1:] - values[:-1]) / (r_here[1:] - r_here[: -1])


def analytical_solution(time_list, c, r_0, viscosity_at_r_0, gamma, grid):
    solution_list = []
    ts = r_0 ** 2 / (3. * (2. - gamma) ** 2 * viscosity_at_r_0)
    # self.viscosity_time_scale = ts
    for time in time_list:
        T = time / ts + 1.
        solution = c * np.exp(-1. * np.power(grid.r / r_0, 2 - gamma) / T) * np.power(T, -1 * (
                2.5 - gamma) / (2 - gamma)) / (
                           3. * np.pi * viscosity_at_r_0 * (np.power(grid.r / r_0, gamma)))
        solution_list.append(solution)
    return solution_list


def get_flux(dt, x, x_i, rho, v_i):  # , phi=flux_limiters.phi_donor_cell):
    """
    this function is from Dr.TIL

    Calculate the flux at the cell interfaces.

    Arguments:
    ----------
    x : array
        cell centers

    x_i : array
        cell interfaces

    rho : array
        density on the grid centers

    v_i : array
        velocity on the grid interfaces

    Keywords:
    ---------

    phi : function
        flux limiter function phi(r)
    """
    f_i = np.zeros_like(x_i)
    r = np.zeros_like(x_i)
    r_minus = np.zeros_like(x_i)

    floor = 1e-50

    # calculate r as in Eq. 4.37

    r[2:-2] = (rho[1:-2] - rho[:-3]) / (rho[2:-1] - rho[1:-2] + floor) * (x[2:-1] - x[1:-2]) / (x[1:-2] - x[:-3])
    r_minus[2:-2] = (rho[3:] - rho[2:-1]) / (rho[2:-1] - rho[1:-2] + floor) * (x[2:-1] - x[1:-2]) / (x[3:] - x[2:-1])
    mask = v_i < 0
    r[mask] = r_minus[mask]

    # calculate the flux limiter function

    # _phi = phi(r)
    _phi = np.maximum(0, np.maximum(np.minimum(1, 2 * r), np.minimum(2, r)))

    # calculate the flux, Eq. 4.38

    f_i[2:-2] = np.maximum(0.0, v_i[2:-2]) * rho[1:-2] + np.minimum(0.0, v_i[2:-2]) * rho[2:-1] + \
                0.5 * abs(v_i[2:-2]) * (
                        1.0 - abs(v_i[2:-2]) * dt / (0.5 * (np.sign(v_i[2:-2]) + 1.0) * (x_i[2:-2] - x[1:-2]) + 0.5 * (
                        np.sign(v_i[2:-2]) - 1.0) * (x_i[3:-1] - x[2:-1]) + floor)
                ) * _phi[2:-2] * (rho[2:-1] - rho[1:-2])
    return f_i


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

####################################
#
#	Gas Properties
#
####################################

def SoundSpeed(Tc):
    return np.sqrt(R * Tc / mu)


def KeplerianFrequency(grid, star_mass):
    return np.sqrt(G * star_mass / grid.r ** 3.)


def ScaleHeight(SoundSpeed, KeplerianFrequency):
    return SoundSpeed / KeplerianFrequency


# Visicocity Related Part


def Viscosity_Alpha(alpha, SoundSpeed, KeplerianFrequency):
    return alpha * SoundSpeed ** 2 / KeplerianFrequency


def ViscousVelocityArrays(grid, SurfaceDensity, Viscosity):
    SurfaceDensity_05 = 0.5 * (SurfaceDensity[:-1] + SurfaceDensity[1:])
    u_r_raw = (-3. / (SurfaceDensity_05 * np.sqrt(grid.r_05) + FLOOR)) * derivative(
        Viscosity * SurfaceDensity * np.sqrt(grid.r), grid.r)
    u_r_interface = interpolate.interp1d(grid.r_05, u_r_raw, kind="linear", fill_value="extrapolate")(grid.r_interface)
    u_r = interpolate.interp1d(grid.r_05, u_r_raw, kind="linear", fill_value="extrapolate")(grid.r)
    sign_u_r_interface = np.sign(u_r_interface)
    return u_r, u_r_interface, sign_u_r_interface


def AdvectionFlux(grid, q, u_r_interface, dt, flux_limiter):
    dQ = np.zeros_like(q)
    sign_u_r_interface = np.sign(u_r_interface)
    phi, r_value = np.zeros_like(grid.r), np.zeros_like(grid.r)
    if flux_limiter is 'donner_cell':
        flux = 0.5 * u_r_interface[1:-1] * (
                (1 + sign_u_r_interface[1:-1]) * q[:-1] + (1 - sign_u_r_interface[1:-1]) * q[
                                                                                           1:])  # do have length nr+2*n_ghost-1
    else:
        for ir in range(grid.n_ghost, grid.n_grid + grid.n_ghost, 1):
            if u_r_interface[ir] >= 0.:
                r_value[ir] = (q[ir - 1] - q[ir - 2]) / (q[ir] - q[ir - 1] + FLOOR)
            elif u_r_interface[ir] <= 0.:
                r_value[ir] = (q[ir + 1] - q[ir]) / (q[ir] - q[ir - 1] + FLOOR)
        if flux_limiter is 'minmod':
            phi = np.minimum(1., r_value)
        elif flux_limiter is 'Fromm':
            phi = 0.5 * (1 + r_value)
        elif flux_limiter is 'Beam_Warming':
            phi = r_value
        elif flux_limiter is 'MC':
            phi = np.maximum(0, np.minimum((1 + r_value) / 2, 2, 2 * r_value))
        elif flux_limiter is 'superbee':
            # phi = np.zeros_like(self.r_value)
            phi = np.maximum(0, np.maximum(np.minimum(1, 2 * r_value), np.minimum(2, r_value)))
        elif flux_limiter is 'van_Leer':
            phi = (r_value + np.abs(r_value)) / (1 + np.abs(r_value))
        elif flux_limiter is 'Lax_Wendroff':
            phi = np.ones_like(r_value)
        else:
            print('Invalid flux Limiter String!')

        flux = 0.5 * u_r_interface[1:-1] * (
                (1 + sign_u_r_interface[1:-1]) * q[:-1] + (1 - sign_u_r_interface[1:-1]) * q[1:])

        flux += 0.5 * np.abs(u_r_interface[1:-1]) * (
                1 - np.abs(u_r_interface[1:-1] * dt / (FLOOR + (0.5 * (sign_u_r_interface[
                                                                       1:-1] + 1.) * (
                                                                        grid.r_interface[
                                                                        1:-1] - grid.r[
                                                                                :-1]) + 0.5 * (
                                                                        sign_u_r_interface[
                                                                        1:-1] - 1.) * (
                                                                        grid.r_interface[
                                                                        2:] - grid.r[
                                                                              1:]))))) * phi[
                                                                                         1:] * (
                        q[1:] - q[:-1])

    dQ[1:-1] = dt * (flux[:-1] - flux[1:]) / (grid.r_interface[2:-1] - grid.r_interface[1:-2])
    return dQ
    # q += self.delta_q1_a
    # self.ALayer[1:-1] = q[1:-1] / self.grid.r[1:-1]


def RateWindMHD(SurfaceDensity, Omega):
    return -1 * 2e-5 * Omega * SurfaceDensity / math.sqrt(2 * math.pi)


def do_simulate(gas, flux_limiter, CFL, t_end):
    i = 0
    time_start = datetime.datetime.now()
    print('this Simulation started at time : {}'.format(time_start))
    while gas.time_now <= t_end:
        # gas.one_step_forward_diffusive(CFL)
        gas.OneStepForward(flux_limiter, CFL)

        if gas.time_now > snapshot_time_point[i]:
            gas.take_snapshot()
            print(
                'this is: ' + str(
                    i) + 'th snapshot | {:7.3f}% simulated | time consumed: {} | {} remain estimated!'.format(
                    gas.time_now * 100 / t_end, datetime.datetime.now() - time_start,
                    (datetime.datetime.now() - time_start) / (gas.time_now * 100 / t_end) * (
                            100 - gas.time_now * 100 / t_end)))
            i += 1


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

class Grid:
    def __init__(self, n_grid, n_ghost, mode, r_min_0, r_max_0):
        self.n_grid = n_grid
        self.n_ghost = n_ghost
        self.mode = mode
        if mode is 'linear':
            self.dr = (r_max_0 - r_min_0) / n_grid
            self.r_interface = np.linspace(r_min_0 - (self.n_ghost + 0.5) * self.dr,
                                           r_max_0 + (self.n_ghost + 0.5) * self.dr,
                                           num=n_grid + 2 * n_ghost + 1, endpoint=True)
        elif mode is 'log':
            self.dr_log = (np.log(r_max_0) - np.log(r_min_0)) / n_grid
            self.r_interface = np.logspace(math.log(r_min_0) - (self.n_ghost + 0.5) * self.dr_log,
                                           math.log(r_max_0) + (self.n_ghost + 0.5) * self.dr_log,
                                           num=n_grid + 2 * n_ghost + 1,
                                           base=np.e,
                                           endpoint=True)
        self.r = 0.5 * (self.r_interface[:-1] + self.r_interface[1:])
        self.r_min = self.r[0]
        self.r_max = self.r[-1]
        self.r_05 = 0.5 * (self.r[0:-1] + self.r[1:])
        self.OK = True

    def __repr__(self):
        if self.OK:
            return str(self.n_grid) + ' ' + self.mode + ' grids has been set.'
        else:
            return 'Grids set up FAILURE.'


class LayeredGas:
    def __init__(self, star_mass, grid, alpha_active, alpha_dead, initial_density, initial_temperature_profile,
                 inner_boundary, outer_boundary, model):
        self.time_now = 0.
        self.model = model
        self.star_mass = star_mass
        self.grid = grid
        self.inner_boundary = inner_boundary
        self.outer_boundary = outer_boundary
        self.InitialDensity = initial_density
        self.SurfaceDensity = initial_density
        self.alpha_a = alpha_active
        self.alpha_d = alpha_dead
        self.Tc = initial_temperature_profile
        self.Omega_K = KeplerianFrequency(self.grid, self.star_mass)
        self.sound_speed = SoundSpeed(self.Tc)
        self.ALayer = np.zeros_like(self.SurfaceDensity)
        self.Ddzone = np.zeros_like(self.SurfaceDensity)
        # self.viscosity_a = np.zeros_like(self.grid.r)
        # self.viscosity_d = np.zeros_like(self.grid.r)
        self.viscosity_a = Viscosity_Alpha(self.alpha_a, self.sound_speed, self.Omega_K)
        self.viscosity_d = Viscosity_Alpha(self.alpha_d, self.sound_speed, self.Omega_K)

        self.u_r_a, self.u_r_interface, self.sign_u_r_interface = np.zeros_like(self.grid.r), np.zeros_like(
            self.grid.r_interface), np.zeros_like(self.grid.r_interface)
        self.SurfaceDensity_snapshots = []

    def update(self):
        if self.model in [2, 4]:
            Layer_is_or_no = self.SurfaceDensity > 2. * 7.5
            for i in range(len(self.SurfaceDensity)):
                if Layer_is_or_no[i]:
                    self.ALayer[i] = 2. * 7.5
                    self.Ddzone[i] = self.SurfaceDensity[i] - 2. * 7.5
                else:
                    self.ALayer[i] = self.SurfaceDensity[i]
                    self.Ddzone[i] = 0.
            # self.alpha_d = (1 - np.tanh((self.InitialDensity - 15.) / (1.) * 0.2)) * 2.e-2 / 2. + 1.e-4
        elif self.model in [1, 3]:
            self.alpha_a = (1 - np.tanh((self.InitialDensity - 15.) / (1.) * 0.2)) * 2.e-2 / 2. + 1.e-5    #1e-2  # 
            self.ALayer = self.SurfaceDensity
            self.Ddzone = np.zeros_like(self.Ddzone)
        elif self.model in [10,11]:
            Layer_is_or_no = self.SurfaceDensity > 2. * 7.5
            for i in range(len(self.SurfaceDensity)):
                if Layer_is_or_no[i]:
                    self.ALayer[i] = 2. * 7.5
                    self.Ddzone[i] = self.SurfaceDensity[i] - 2. * 7.5
                else:
                    self.ALayer[i] = self.SurfaceDensity[i]
                    self.Ddzone[i] = 0.
            
            self.alpha_a = (2e-2 * self.ALayer + self.alpha_d * self.Ddzone)/self.SurfaceDensity
            self.ALayer = self.SurfaceDensity
            self.Ddzone = np.zeros_like(self.Ddzone)
        else:
            pass

        self.viscosity_a = Viscosity_Alpha(self.alpha_a, self.sound_speed, self.Omega_K)
        self.viscosity_d = Viscosity_Alpha(self.alpha_d, self.sound_speed, self.Omega_K)
        self.u_r_a, self.u_r_interface_a, self.sign_u_r_interface_a = ViscousVelocityArrays(self.grid, self.ALayer,
                                                                                            self.viscosity_a)
        self.u_r_d, self.u_r_interface_d, self.sign_u_r_interface_d = ViscousVelocityArrays(self.grid, self.Ddzone,
                                                                                            self.viscosity_d)

    def apply_boundary_conditions(self):
        if self.inner_boundary is 'ZG':
            # self.SurfaceDensity[:self.grid.n_ghost] = self.SurfaceDensity[self.grid.n_ghost + 1]
            self.ALayer[:self.grid.n_ghost] = self.ALayer[self.grid.n_ghost + 1]
            self.Ddzone[:self.grid.n_ghost] = self.Ddzone[self.grid.n_ghost + 1]

        elif self.inner_boundary is 'floor':
            # self.SurfaceDensity[: self.grid.n_ghost] = FLOOR
            self.ALayer[: self.grid.n_ghost] = FLOOR
            self.Ddzone[: self.grid.n_ghost] = FLOOR

        if self.outer_boundary is 'floor':
            # self.SurfaceDensity[-self.grid.n_ghost:] = FLOOR
            self.ALayer[-self.grid.n_ghost:] = FLOOR
            self.Ddzone[-self.grid.n_ghost:] = FLOOR
        elif self.outer_boundary is 'hello':
            pass

    def OneStepForward(self, flux_limiter, parameter):
        self.update()
        self.apply_boundary_conditions()
        dt_a = parameter * np.amin(
            ((self.grid.r_interface[1:] - self.grid.r_interface[:-1]) / (self.sound_speed + np.abs(
                self.u_r_interface_a[:-1])))[2:-2])
        dt_d = parameter * np.amin(
            ((self.grid.r_interface[1:] - self.grid.r_interface[:-1]) / (self.sound_speed + np.abs(
                self.u_r_interface_d[:-1])))[2:-2])
        dt = min(dt_a, dt_d)
        self.ALayer += AdvectionFlux(self.grid, self.ALayer * self.grid.r, self.u_r_interface_a, dt,
                                     flux_limiter) / self.grid.r
        self.Ddzone += AdvectionFlux(self.grid, self.Ddzone * self.grid.r, self.u_r_interface_d, dt,
                                     flux_limiter) / self.grid.r
        if self.model in [3, 4, 11]:
            self.SurfaceDensity = self.ALayer + self.Ddzone + dt * RateWindMHD(self.SurfaceDensity, self.Omega_K)
        else:
            self.SurfaceDensity = self.ALayer + self.Ddzone
        self.time_now += dt

    def take_snapshot(self):
        self.SurfaceDensity_snapshots.append(np.copy(self.SurfaceDensity))


############################################################################################################################

gas_grid = Grid(300, 1, 'linear', 1 * Au, 200 * Au)
r_0 = Au
Tc_0 = 130 * np.power(gas_grid.r / Au, -0.5)
# viscosity_at_r_0 = 0.01 * np.sqrt(R * 130 * np.power(1, -0.5) / mu) ** 2 / np.sqrt(G * 1 * M_sun / Au ** 3.)
# viscosity_time_scale = r_0 ** 2 / (3. * (2. - gamma) ** 2 * viscosity_at_r_0)  # r_0 ** 2 / viscosity_at_r_0
snapshot_time_point = 1e6 * Year * np.array([0, 0.1, 0.5, 1, 5])  #np.linspace(0, 5, 20, endpoint=True)  # 
Sigma_0 = 6.
Sigma_initial = Sigma_0 * (gas_grid.r / (100. * Au)) ** -1
gasL = LayeredGas(star_mass=1. * M_sun, grid=gas_grid, alpha_active=2e-2, alpha_dead=1e-5,
                  initial_density=Sigma_initial, initial_temperature_profile=Tc_0,
                  inner_boundary='floor', outer_boundary='floor', model=10)
FluxLimiter = ['donner_cell', 'minmod', 'Fromm', 'Beam_Warming', 'MC', 'superbee', 'van_Leer', 'Lax_Wendroff']
FL_here = FluxLimiter[0]
do_simulate(gasL, FL_here, .5, snapshot_time_point[-1])

