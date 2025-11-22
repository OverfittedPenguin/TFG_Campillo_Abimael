import numpy as np
import os
import casadi as ca
import matplotlib as mpl
import matplotlib.pyplot as plt
from aircraft import Aircraft
from simulation import Sim
from atmosphere import Atmos
from nonlinearprogramming import NLP_CRUISE

###########################################################
##                  USER CONFIGURATION                   ##
###########################################################

# SIMULATION CONFIG FILE PATH
simulation_file = "configs/Simulation.json"

# ATMOSPHERE CONFIG FILE PATH
atmos_file = "configs/Atmos.json"

###########################################################
##                  PREPROCESS OF FILES                  ##
###########################################################

if not os.path.isfile(simulation_file):
    raise FileNotFoundError(f"Simulation JSON file not found at: {simulation_file}")
if not os.path.isfile(atmos_file):
    raise FileNotFoundError(f"Atmosphere JSON file not found at: {atmos_file}")

sim = Sim.from_json(simulation_file)
atmos = Atmos.from_json(atmos_file)

# AIRCRAFT CONFIG FILE PATH
aircraft_file = sim.Aircraft_file

# Validate that files exist.
if not os.path.isfile(aircraft_file):
    raise FileNotFoundError(f"Aircraft JSON file not found at: {aircraft_file}")

aircraft = Aircraft.from_json(aircraft_file)

print("AIRCRAFT LOADED:", aircraft.name)
print("ATMOS CONDITIONS LOADED.")
print("SIMULATION CONDITIONS LOADED: dT=",sim.dT,"tF=", sim.tF)

###########################################################
##          PROBLEM DEFINITION AND SOLUTION              ##
###########################################################

# PREPROCESS

# CRUISE SUBPROBLEM DEFINITION. Cruise flight trajectory
# defined as a NLP problem.
nlp = NLP_CRUISE()
w0, w, lbx, ubx, g, lbg, ubg = NLP_CRUISE.CONSTRAINTS_AND_BOUNDS(nlp.x,nlp.u,aircraft,atmos,sim)
J = NLP_CRUISE.COST_FUCNTIONAL(w,aircraft,atmos,sim)

# Redefining vectors as stipulated by CASADi dictionary.
w0 = ca.vertcat(w0)
w = ca.vertcat(w)
g = ca.vertcat(*g)

# SOLVER

# Configuration of the NLP and the solver.
nlp = {"x": w, "f": J, "g": g}
solver = ca.nlpsol("solver", "ipopt", nlp)

# TRAJECTORIES. SOLUTION
sol = solver(
    x0 = w0,
    lbx = lbx,
    ubx = ubx,
    lbg = lbg,
    ubg = ubg
)

###########################################################
##                     POSTPROCESS                       ##
###########################################################

# RECONSTRUCTION OF STATE VECTOR THROUGH TIME
x = sol['x'].full().flatten()
t = np.linspace(0.0, sim.tF, sim.N)

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
u1 = []
u2 = []

for k in range(sim.N):
    idx = 9*k
    x1.append(x[idx])
    x2.append(x[idx + 1])
    x3.append(x[idx + 2])
    x4.append(x[idx + 3])
    x5.append(x[idx + 4])
    x6.append(x[idx + 5])
    x7.append(x[idx + 6])
    u1.append(x[idx + 7])
    u2.append(x[idx + 8])

# Convertir a numpy arrays
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
x5 = np.array(x5)
x6 = np.array(x6)
x7 = np.array(x7)
u1 = np.array(u1)
u2 = np.array(u2)

# Calcular alpha
alpha = np.arctan2(x2 - sim.wind[1], x1 - sim.wind[0])

# PLOTS
plt.plot(t,x1)
plt.plot(t,x2)
plt.show()

plt.plot(t,x3)
plt.show()

plt.plot(t,x4)
plt.plot(t,alpha)
plt.show()

plt.plot(t,x5)
plt.plot(t,-x6)
plt.show()

plt.plot(t,x7)
plt.show()

plt.plot(t,u1)
plt.show()

plt.plot(t,u2)
plt.show()

plt.plot(x5,-x6)
