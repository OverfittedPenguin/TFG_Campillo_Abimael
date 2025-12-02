import numpy as np
import os
import casadi as ca
from aircraft import Aircraft
from simulation import Sim
from atmosphere import Atmos
from nonlinearprogramming import NLP_CRUISE
import matplotlib.pyplot as plt

###########################################################
##                 USER CONFIGURATION                    ##
###########################################################

# SIMULATION CONFIG FILE PATH
simulation_file = "configs/Simulation.json"

# ATMOSPHERE CONFIG FILE PATH
atmos_file = "configs/Atmos.json"

###########################################################
##            PREPROCESS OF FILES AND VARS               ##
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

# N, number of nodes. tF, end time.
N = np.linspace(25,500)
N = N.astype(int)

sim.tF = 60.0

###########################################################
##          PROBLEM DEFINITION AND SOLUTION              ##
###########################################################

# OBJECTIVE STORAGE
obj =[]

for k in range(len(N)):
    # Initial states.
    # Initial mass.
    sim.x0[7] = aircraft.BEM + aircraft.FM + aircraft.PM

    # Computation of trim conditions for controls at initial state.
    x0 = sim.x0[1:8]

    # Initial state vector
    sim.w0 = x0

    # Time-step assignation.
    sim.N = N[k]
    sim.dT = sim.tF / sim.N

    # CRUISE SUBPROBLEM DEFINITION. Cruise flight trajectory
    # defined as a NLP problem.
    nlp = NLP_CRUISE()
    w0, w, lbx, ubx, g, lbg, ubg = NLP_CRUISE.CONSTRAINTS_AND_BOUNDS(nlp.x,nlp.u,aircraft,atmos,sim)
    J = NLP_CRUISE.COST_FUCNTIONAL(w,aircraft,atmos,sim)

    # Redefining vectors as stipulated by CASADi dictionary.
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

    # Retrieving of iterations values and objective value.
    obj.append(solver.stats()['iterations']['obj'][-1])

###########################################################
##                     POSTPROCESS                       ##
###########################################################

# PLOTS
path = os.path.join(os.getcwd(), "images", "convergence")
os.makedirs(path, exist_ok=True)

plt.figure(figsize=(12,8))
plt.subplots_adjust(
    left=0.10, 
    bottom=0.20, 
    right=0.85, 
    top=0.85
    )

# Cost plot.
plt.scatter(N,obj,label="Final cost J",color='#004D99',linestyle="-",linewidth=1.5)

# Titles, grid and legend.
plt.xlabel("Nº of nodes [-]",fontsize = 14,fontstyle='italic',fontfamily='serif')
plt.ylabel("Final cost objective [-]",fontsize = 14, fontstyle='italic', fontfamily='serif')
plt.title(f"Final objective evolution per time-step size",fontsize = 16, fontweight='bold', fontfamily='serif', loc="left")
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.75, color='gray', alpha=0.75)
plt.legend(fontsize=10, prop={'family': 'serif'}, loc="upper left", bbox_to_anchor=(1.02,1))
plt.savefig(os.path.join(path, "COST_vs_N.svg"))
plt.show()

