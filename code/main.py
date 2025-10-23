import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

def geopotential_altitude(h_geometric):
    # Convert geometric altitude to geopotential altitude (m)
    Re = 6371000.0
    return Re * h_geometric / (Re + h_geometric)

def isa_atmosphere(h_m):
    """
    Return (T, P, rho) at geometric altitude h_m (meters) using
    the troposphere ISA (0–11 km).
    """
    # Clip to valid range for this use-case
    h_m = float(np.clip(h_m, 0.0, 11000.0))

    # Constants
    g0 = 9.80665        # m/s^2
    R = 287.053         # J/(kg K)
    P0 = 101325.0       # Pa
    Temp_0 = 288.15     # K
    L = -0.0065         # K/m (troposphere lapse rate, negative)

    # Use geopotential altitude for small precision improvement
    h_geo = geopotential_altitude(h_m)

    Temp = Temp_0 + L * h_geo
    P = P0 * (Temp / Temp_0) ** (-g0 / (R * L))
    rho = P / (R * Temp)
    return Temp, P, rho

def isa_density(h_m):
    """Return density (kg/m^3) at geometric altitude h_m (meters)."""
    return isa_atmosphere(h_m)[2]

def propulsive_forces_moments(rho, V, n, Dp, nEng, CTx, CTz, Coeffs_Ct, eps0, AoA):
    """Return thrust T (N) and pitching thrust coefficient (Cm_dt) for given density, speed, rpm,
    propeller dimensions and thrust coefficients."""

    n_RPS = n / 60.0      # RPS
    J = V / (n * Dp + 1e-9)        # Advance ratio (-), avoid div-by-zero
    eps = eps0 - AoA    # Thrust AoA (deg)

    # Thrust coefficient and force
    Ct = Coeffs_Ct[0] + Coeffs_Ct[1] * J + Coeffs_Ct[2] * J**2 + Coeffs_Ct[3] * J**3
    T = nEng * rho * n_RPS**2 * Dp**4 * Ct

    # Pitching moment due to thrust and adimensional coefficient
    M_dt = -T * CTx * np.sin(np.radians(eps)) - T * CTz * np.cos(np.radians(eps))
    Cm_dt = M_dt / (rho * n_RPS**2 * Dp**5 + 1e-12)
    return T, Cm_dt

# -------------------------
# PARAMETERS
# -------------------------
switch_flaps = "F0"   # Default (flaps up)

# Environmental conditions
g = 9.81            # gravitational acceleration (m/s^2)
wind_speed = [0, 0] # wind speed vector (m/s)

# Aircraft parameters (USER INPUTS)
bem = 20.0        # basic empty mass (kg)
fm = 0.0          # fuel mass (kg)
pm = 0.0        # payload mass (kg)
SFC = 0.0          # specific fuel consumption (kg/(k·Wh))
AR = 7.0            # aspect ratio
S = 0.778            # wing area (m²)
b = 2.33            # wingspan (m)  
c = 0.33             # mean aerodynamic chord (m)
e = 1.78*(1 - 0.045 * AR**0.68) - 0.64  # Oswald efficiency factor (Raymer's correlation,1999)
print(e)
K = 1 / (np.pi * AR * e)  # induced drag factor

CGx = -0.114        # CG position from datum (horizontal, +body X) (m)
CGz = -0.0019        # CG position from datum (verticak, +body Z) (m)
CPwx = -0.07717       # CP wing position from datum (horizontal, +body X) (m)
CPwz = -0.176       # CP wing position from datum (vertical, +body Z) (m)
DeltaX_WG = CPwx - CGx
DeltaZ_WG = CPwz - CGz

Iyy = 3.99          # body Y inertia (kg·m^2)

# Aerodynamic coefficients
if switch_flaps == "F0":
    CL_0 = 0.410
    CL_alpha = 3.650
    CD_0 = 0.068
elif switch_flaps == "F15":
    CL_0 = 0.510
    CL_alpha = 4.584
    CD_0 = 0.085
elif switch_flaps == "F40":
    CL_0 = 0.700
    CL_alpha = 6.303
    CD_0 = 0.112

CL_de = 0.6

# Pitching moment coefficients
Cm_0 = -0.079
Cm_alpha = -1.430
Cm_de = -2.160

# Propulsion parameters
Dp = 0.381         # Proppeler's diameter (m)
nEng = 2.0         # Number of engines (-)
CTx = 0.000        # CT position form datum (horizontal, +body X) (m)
CTz = -0.176      # CT position from datum (vertical, +body Z) (m)
DeltaX_T = CTx - CGx
DeltaZ_T = CTz - CGz
eps0 = 0.0         # Thrust line angle respect to fuselage center line (deg) 

# -------------------------
# 2D DYNAMICS
# -------------------------
def eom_3dof(t, state, ctrls=None, trimpars=None):

    if ctrls is None and trimpars is None:              # For Propagation
        ctrls = control                          # use runtime control dict (defined later)
        theta_dot_trim = 0.0
        wsp = wind_speed

        # expect full navigation state: [u, w, q, theta, x, z]
        u, w, q, theta, x, z = state[0:6]

    elif ctrls is not None and trimpars is not None:    # For Trimming
        V_trim, gamma_trim, theta_dot_trim = trimpars
        wsp = np.array([0.0, 0.0])               # no wind in trim
        # trim state vector: [u, w, q, theta]
        u, w, q, theta = state[0:4]

    else:
        raise ValueError("Wrong dynamic inputs... ")

    # Windspeed effect:
    ua, wa = u - wsp[0], w - wsp[1]

    # Aerodynamic variables:
    V = np.sqrt(u**2 + w**2)
    Va = np.sqrt(ua**2 + wa**2)
    alpha = np.arctan2(wa, ua)   # robust AoA
    rho = isa_density(z if len(state) >= 6 else 0.0)
    q_bar = 0.5 * rho * Va**2

    # Nondimensional pitch rate
    q_ast = q * c / (2.0 * max(Va, 1e-6))

    # Aerodynamic forces:
    CL = CL_0 + CL_alpha * alpha + CL_de * ctrls['elevator']
    CD = CD_0 + K * CL**2

    # Propulsive model (placeholder values used if not calling real prop routine)
    Tmax = 120.0
    Cm = Cm_0 + Cm_alpha * alpha + Cm_de * ctrls['elevator']

    # Propulsive force
    T = ctrls['throttle'] * Tmax
    Fb_prop = np.array([T, 0.0])

    # Rotation matrices
    Rsb = np.array([  # body to stability (2D)
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])
    Rhb = np.array([  # body to inertial (horizontal) (2D)
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    # Mass
    mass = bem + fm + pm

    # Aerodynamic forces in body axes
    Cb_aero = Rsb.T @ np.array([-CD, -CL])
    Fb_aero = q_bar * S * Cb_aero

    # Gravity in inertial frame is downward -> [0, -mass*g]; transform to body
    Fb_grav = Rhb.T @ np.array([0.0, mass * g])

    # Moments (scalar in 2D, pitching about Y)
    Mb_aero = q_bar * S * c * Cm
    Mb_prop =  -T*DeltaZ_T
    Mb_total = Mb_aero + Mb_prop

    # Total forces
    Fb_total = Fb_aero + Fb_prop + Fb_grav

    # Equations of motion
    u_dot = Fb_total[0] / mass - w*q
    w_dot = Fb_total[1] / mass + u*q
    q_dot = Mb_total / Iyy 

    # Kinematics
    theta_dot = q - theta_dot_trim

    # Navigation (inertial velocities)
    inertial_vel = Rhb @ np.array([u, w])
    x_dot, z_dot = inertial_vel

    if ctrls is not None and trimpars is not None:
        # Constraints for trimming
        ap = np.cos(alpha)
        bp = np.sin(alpha)
        roc_dot = np.sin(gamma_trim) - ap * np.sin(theta) + bp * np.cos(theta)
        sc_dot = V_trim - V

        return [
            u_dot, w_dot,
            q_dot, theta_dot,
            roc_dot, sc_dot,
        ]
    else:
        return [
            u_dot, w_dot,
            q_dot, theta_dot,
            x_dot, z_dot
        ]

def trim_residuals(trim_vars):
    """
    Function to compute the residuals (i.e., derivatives) for trim conditions.
    """
    # Extract state and control from trim_vars
    # System states (without navigation states) and controls
    u, w, q, theta, delta_t, delta_e = trim_vars
    
    # Construct full state vector (for steady, symmetric flight)
    state_trim = np.array([
        u,       # u
        w,       # w
        q,       # q
        theta,   # theta
    ])
    
    # Control inputs (example: elevator and throttle)
    controls_trim = {
        'throttle': delta_t,    # delta_t [0-1]
        'elevator': delta_e,    # delta_e [rad]
    }

    # Get derivatives from EOM
    dydt = eom_3dof(0, state_trim, controls_trim, trimpars)
    
    # Return the relevant derivatives (residuals) we want to drive to zero
    return np.array([
        dydt[0],  # u_dot ≈ 0
        dydt[1],  # w_dot ≈ 0
        dydt[2],  # q_dot ≈ 0
        dydt[3],  # theta_dot ≈ 0
        dydt[4],  # roc_dot ≈ 0
        dydt[5],  # sc_dot ≈ 0
    ])

# -----------------------------
# Trimming
# -----------------------------

# Trim constraints:
trimpars = np.array([40,                    # [m/s] V_trim (desired speed),  # maqueta 46 kts
                     np.deg2rad(0.0),       # [deg -> rad] gamma_trim (flight path angle, rate-of-climb), 
                     np.deg2rad(0.0)])      # [deg -> rad] theta_dot_trim (pitch angle speed, pull-up),

# Initial guess: [state_trim, control_trim]
trim_guess = [trimpars[0], 0.0,    # u,w
              0.0, 0.0,            # q,theta
              0.5, 0.0,]           # delta_t,delta_e

# Solve Trim (Equilibrium == 0 derivatives)
result = root(trim_residuals, trim_guess)
result.x[3] = (result.x[3] + np.pi) % (2 * np.pi) - np.pi       # angle wrapping for euler [-pi,+pi]
residuals = trim_residuals(result.x)                            # should be numerically small

if result.success:
    print("Trim condition found:")
    print(result.x)
    print(f"u = {result.x[0]:.3f} m/s")
    print(f"w = {result.x[1]:.3f} m/s")
    print(f"q = {result.x[2]:.3f} rad/s")
    print(f"theta = {result.x[3]:.3f} rad")
    print(f"delta_t = {result.x[4]:.3f} ---")
    print(f"delta_e = {result.x[5]:.3f} rad")
    print("Trim derivatives:")
    print(residuals)
    if np.max(np.abs(residuals)) > 1e-3:
        print("WARNING: Large derivative - unfeasible trim... REVIEW!")
else:
    print("Trim solver failed:", result.message) 

# -----------------------------
# Simulation
# -----------------------------

# Initial State

# [u, w, q, theta, x, z]
state0 = np.array([
    result.x[0], result.x[1],      # velocities
    result.x[2], result.x[3],      # angular rate and euler angle
    0.0, -1000.0  # position
])

# Control input
control = {
    'throttle': result.x[4],    # delta_t [0-1] +- 0.1
    'elevator': result.x[5],   # delta_e [rad] +- 0.01 (very sensitive / powerful tail)
}

sol = solve_ivp(
    fun=eom_3dof,
    t_span=(0, 100),
    y0=state0,
    t_eval=np.linspace(0, 100, 1000),
    method='BDF', # 'RK45' 'Radau' 'BDF'
)

# -----------------------------
# Plotting
# -----------------------------

t = sol.t
u, w = sol.y[0], sol.y[1]
q = sol.y[2]
theta = sol.y[3]
x, z = sol.y[4], sol.y[5]

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(t, u, label="u")
plt.plot(t, w, label="w")
plt.title("Body Velocities (m/s)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, np.degrees(theta), label="Pitch")
plt.title("Euler Angles (degrees)")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, np.degrees(q), label="q")
plt.title("Angular Rates (degrees/s)")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, x, label="x")
plt.plot(t, z, label="z")
plt.title("Position (m)")
plt.legend()
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
