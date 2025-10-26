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

# Initial state: [u, w, q, theta, x, z]
state0 = np.array([22.5, 0.0, 0.0, 0.0, 0.0, -500.0])

# Define flight phases
phases = [
    ("DESCENT", 240, ctrls := {"elevator": -0.0686, "throttle": 0.35}), # descent for 80 seconds
    ("CRUISE", 120,ctrls := {"elevator": -0.058, "throttle": 0.67}), # cruise for 100 seconds
    ("CLIMB", 180, ctrls := {"elevator": -0.08, "throttle": 1.0}),   # climb for 60 seconds
]

# Storage for concatenated results
t_total = []
X_total = []

# Initial state
current_state = state0
t_start = 0.0

for phase_name,duration,ctrls in phases:
    t_span = (t_start, t_start + duration)
    t_eval = np.linspace(t_span[0], t_span[1], int(duration*100)+1)  # 10 points/sec
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

    # Append results
    t_total.append(sol.t)
    X_total.append(sol.y.T)

    # Update initial state for next phase
    current_state = sol.y[:, -1]
    t_start += duration

# Concatenate arrays
t_total = np.concatenate(t_total)
X_total = np.concatenate(X_total)

# Plot results
plt.figure(figsize=(12, 8))

# Altitude
plt.subplot(3,1,1)
plt.plot(t_total, -X_total[:,5])
#plt.plot(t_total,X_total[:,4])
plt.ylabel("Altitude [m]")
plt.grid(True)

# Pitch angle
plt.subplot(3,1,2)
plt.plot(t_total, X_total[:,3])
plt.ylabel("Pitch [rad]")
plt.grid(True)

# Airspeed
plt.subplot(3,1,3)
V = np.sqrt(X_total[:,0]**2 + X_total[:,1]**2)
plt.plot(t_total, X_total[:,0])
plt.ylabel("Airspeed [m/s]")
plt.xlabel("Time [s]")
plt.grid(True)

plt.tight_layout()
plt.show()