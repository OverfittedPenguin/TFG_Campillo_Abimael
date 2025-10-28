import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from aircraft import Aircraft
from atmosphere import Atmos
from dynamic_equations import DynEqs

# AIRCRAFT CONFIG FILE PATH
aircraft_file = "configs/FlyoxI_VI.json"

# ATMOSPHERE CONFIG FILE PATH
atmos_file = "configs/Atmos.json"

# Validate that files exist.
if not os.path.isfile(aircraft_file):
    raise FileNotFoundError(f"Aircraft JSON file not found at: {aircraft_file}")
if not os.path.isfile(atmos_file):
    raise FileNotFoundError(f"Atmosphere JSON file not found at: {atmos_file}")

# Load aircraftparameters and atmosphere conditions.
aircraft = Aircraft.from_json(aircraft_file)
atmos = Atmos.from_json(atmos_file)
print("AIRCRAFT LOADED:", aircraft.name)
print("ATMOS CONDITIONS LOADED.")

# Initial state: [u, w, q, theta, x, z].
state0 = np.array([22.5, 0.0, 0.0, 0.0, 0.0, -500.0])

# Define flight phases.
phases = [
    ("DESCENT", 240, ctrls := {"elevator": -0.0686, "throttle": 0.35}), # descent for 80 seconds
    ("CRUISE", 120,ctrls := {"elevator": -0.058, "throttle": 0.67}), # cruise for 100 seconds
    ("CLIMB", 180, ctrls := {"elevator": -0.08, "throttle": 1.0}),   # climb for 60 seconds
]

# Storage for concatenated results.
t_total = []
X_total = []
ctrls_T_total = []

# Initial state.
current_state = state0
t_start = 0.0

for phase_name,duration,ctrls in phases:
    t_span = (t_start, t_start + duration)
    t_eval = np.linspace(t_span[0], t_span[1], int(duration*100)+1)
    ctrls_T = np.ones(t_eval.size)*ctrls['throttle']
    dyn = DynEqs(aircraft, atmos,ctrls)
    sol = solve_ivp(
        lambda t, y: dyn.EOM_3DOF(t, y, phase_name),
        t_span,
        current_state,
        method='Radau',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )

    # Append results.
    t_total.append(sol.t)
    X_total.append(sol.y.T)
    ctrls_T_total.append(ctrls_T)

    # Update initial state for next phase.
    current_state = sol.y[:, -1]
    t_start += duration

# Concatenate arrays.
t_total = np.concatenate(t_total)
X_total = np.concatenate(X_total)
ctrls_T_total = np.concatenate(ctrls_T_total)

# RETRIEVE CONTROL VARIABLES
alpha = np.arctan2(X_total[:,1], X_total[:,0])
V = np.sqrt(X_total[:,0]**2 + X_total[:,1]**2)
T = np.zeros(alpha.size)
m = np.zeros(alpha.size)

for i in range(alpha.size):
    rho, sigma = atmos.ISA_RHO(-X_total[i,5])
    T_max,_,P = aircraft.PROPULSIVE_FORCES_MOMENTS(V[i],6000.0,rho,sigma,alpha[i])
    T[i] = T_max
    m[i] = aircraft.BEM + aircraft.FM * (1 - aircraft.SFC * P * t_total[i]) + aircraft.PM

T = T*ctrls_T_total

# PLOTS
#ALTITUDE AND HORIZONTAL DISTANCE
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(
    t_total, X_total[:, 4],
    color='0.6',
    label='Horizontal dist.',
    linestyle='--',
    linewidth=2
)
ax1.set_xlabel('Time [s]',  color='0', fontstyle='italic', fontsize=12)
ax1.set_ylabel('Horizontal distance [m]', color='0', fontstyle='italic', fontsize=12)
ax1.tick_params(axis='y', labelcolor='0')
ax2 = ax1.twinx()
ax2.plot(
    t_total, -X_total[:, 5],
    color='0.25',
    label='Altitude',
    linestyle='-',
    linewidth=2
)
ax2.set_ylabel('Altitude [m]', color='0', fontstyle='italic', fontsize=12)
ax2.tick_params(axis='y', labelcolor='0')
ax1.set_title('Altitude and distance through time',
              fontstyle='italic', fontsize=14, fontweight='bold',loc='left')
ax1.grid(True)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper right', fontsize=12, fancybox=True, framealpha=1, facecolor='white')
fig.tight_layout()

# PITCH AND AOA
fig2, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(
    t_total, X_total[:, 3],
    color='0.6',
    label='Pitch',
    linestyle='-',
    linewidth=2
)
ax3.plot(
    t_total, alpha,
    color='0.25',
    label='AoA',
    linestyle='-',
    linewidth=2
)
ax3.set_xlabel('Time [s]',  color='0', fontstyle='italic', fontsize=12)
ax3.set_ylabel('Angles [rad]', color='0', fontstyle='italic', fontsize=12)
ax3.tick_params(axis='y', labelcolor='0')
ax3.set_title('Pitch and AoA through time',
              fontstyle='italic', fontsize=14, fontweight='bold',loc='left')
ax3.grid(True)
lines_1, labels_1 = ax3.get_legend_handles_labels()
ax3.legend(lines_1, labels_1,
           loc='upper right', fontsize=12, fancybox=True, framealpha=1, facecolor='white')
fig2.tight_layout()

# MASS AND THRUST
fig3, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(
    t_total, T,
    color='0.6',
    label='Thrust',
    linestyle='-',
    linewidth=2
)
ax4.set_xlabel('Time [s]',  color='0', fontstyle='italic', fontsize=12)
ax4.set_ylabel('Thrust [N]', color='0', fontstyle='italic', fontsize=12)
ax4.tick_params(axis='y', labelcolor='0')
ax4.grid(True)
ax5 = ax4.twinx()
ax5.plot(
    t_total, m,
    color='0.25',
    label='Mass',
    linestyle='--',
    linewidth=2
)
ax5.set_ylabel('Mass [kg]', color='0', fontstyle='italic', fontsize=12)
ax5.tick_params(axis='y', labelcolor='0')
ax4.set_title('Thrust and mass through time',
              fontstyle='italic', fontsize=14, fontweight='bold',loc='left')
ax5.grid(True)
lines_1, labels_1 = ax4.get_legend_handles_labels()
lines_2, labels_2 = ax5.get_legend_handles_labels()
ax5.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper right', fontsize=12, fancybox=True, framealpha=1, facecolor='white')
fig3.tight_layout()
plt.show()