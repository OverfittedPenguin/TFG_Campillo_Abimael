import numpy as np
import os
import casadi as ca
from aircraft import Aircraft
from simulation import Sim
from atmosphere import Atmos
from nonlinearprogramming import NLP_CRUISE
from plotterfunction import Plotter

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

print("AIRCRAFT LOADED:", aircraft.name)
print("ATMOS CONDITIONS LOADED.")
print("SIMULATION CONDITIONS LOADED: dT=",sim.dT,"tF=", sim.tF)

# INITIAL STATE COMPUTATIONS
# Initial mass.
sim.x0[7] = aircraft.BEM + aircraft.FM + aircraft.PM

# Computation of trim conditions for controls at initial state.
x0 = sim.x0[1:8]

# Initial state vector
sim.w0 = x0

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
w = ca.vertcat(w)
g = ca.vertcat(*g)

# SOLVER
# Configuration of the NLP and the solver.
opts = {}
opts['ipopt.max_iter'] = 1000
opts['ipopt.tol'] = 1e-6
opts['ipopt.acceptable_tol'] = 1e-3
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
iters = solver.stats()['iter_count']
obj = solver.stats()['iterations']['obj']
time = np.round(solver.stats()['t_proc_total'],3)

###########################################################
##                     POSTPROCESS                       ##
###########################################################

# RECONSTRUCTION OF STATE VECTOR THROUGH TIME
# Retrieving of states values and time array generation.
x = sol['x'].full().flatten()
t = np.linspace(0.0, sim.tF, sim.N)

# PLOTS
path = os.path.join(os.getcwd(), "images", "benchmark")
os.makedirs(path, exist_ok=True)
Plotter.GENERATE_RESULTS_PLOT(t,x,w0,aircraft,sim,path)
Plotter.GENERATE_COST_PLOT(np.linspace(0,iters,len(obj)),np.array(obj),time,path)
