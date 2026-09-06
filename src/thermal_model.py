"""
Core lumped-parameter thermal network (LPTN) model for the sealed
electronics enclosure.

Six thermal nodes are modeled:
    T1  heater body            T4  interior enclosure wall
    T2  aluminum plate         T5  exterior enclosure wall
    T3  interior air           (T6 is an ambient boundary condition,
                                 not a state variable)

Six resistances are fit to data:
    R_hp    heater -> plate
    R_pa    plate  -> air
    R_aw    air    -> interior wall
    R_pw    plate  -> wall (direct conduction path, bypassing air)
    R_cond  interior wall -> exterior wall
    R_ext   exterior wall -> ambient

Capacitances (C_h, C_p, C_a, C_w, C_e) were measured directly from the
hardware and are held fixed; they are not part of the fit.
"""

import numpy as np
from scipy.integrate import solve_ivp

PARAMETER_NAMES = ["R_hp", "R_pa", "R_aw", "R_pw", "R_cond", "R_ext"]
SENSOR_NAMES = ["T1", "T2", "T3", "T4", "T5"]

# Measured capacitances (J/K), fixed (not fit).
C_h = 48.0
C_p = 75.6
C_a = 0.84
C_w = 60.3
C_e = 60.3

# Per-node weights used when building the least-squares residual vector.
# T3 and T5 are down-weighted slightly to account for higher sensor noise.
NODE_WEIGHTS = {"T1": 1.0, "T2": 1.0, "T3": 0.8, "T4": 1.0, "T5": 0.8}


def thermal_model(t, T, params, P, T_ambient, P_fan=0.0):
    """Right-hand side of the 5-state ODE system (dT/dt)."""
    R_hp, R_pa, R_aw, R_pw, R_cond, R_ext = params
    T1, T2, T3, T4, T5 = T

    dT1 = (P - (T1 - T2) / R_hp) / C_h
    dT2 = ((T1 - T2) / R_hp
           - (T2 - T3) / R_pa
           - (T2 - T4) / R_pw) / C_p
    dT3 = ((T2 - T3) / R_pa
           - (T3 - T4) / R_aw
           + P_fan) / C_a
    dT4 = ((T2 - T4) / R_pw
           + (T3 - T4) / R_aw
           - (T4 - T5) / R_cond) / C_w
    dT5 = ((T4 - T5) / R_cond
           - (T5 - T_ambient) / R_ext) / C_e

    return [dT1, dT2, dT3, dT4, dT5]


def simulate_model(params, time, P, T_init, T_ambient, P_fan=0.0):
    """Integrate the ODE over `time`, returning T(t) with shape (len(time), 5)."""
    sol = solve_ivp(
        lambda t, T: thermal_model(t, T, params, P, T_ambient, P_fan),
        (time[0], time[-1]),
        T_init,
        t_eval=time,
        method="Radau",
        rtol=1e-3,
        atol=1e-6,
    )
    return sol.y.T


def steady_state(params, P, T_ambient):
    """Closed-form steady-state solution (dT/dt = 0), solved algebraically."""
    R_hp, R_pa, R_aw, R_pw, R_cond, R_ext = params

    A = np.zeros((5, 5))
    b = np.zeros(5)

    A[0, 0] = 1 / R_hp
    A[0, 1] = -1 / R_hp
    b[0] = P

    A[1, 0] = -1 / R_hp
    A[1, 1] = 1 / R_hp + 1 / R_pa + 1 / R_pw
    A[1, 2] = -1 / R_pa
    A[1, 3] = -1 / R_pw

    A[2, 1] = -1 / R_pa
    A[2, 2] = 1 / R_pa + 1 / R_aw
    A[2, 3] = -1 / R_aw

    A[3, 1] = -1 / R_pw
    A[3, 2] = -1 / R_aw
    A[3, 3] = 1 / R_pw + 1 / R_aw + 1 / R_cond
    A[3, 4] = -1 / R_cond

    A[4, 3] = -1 / R_cond
    A[4, 4] = 1 / R_cond + 1 / R_ext
    b[4] = T_ambient / R_ext

    return np.linalg.solve(A, b)
