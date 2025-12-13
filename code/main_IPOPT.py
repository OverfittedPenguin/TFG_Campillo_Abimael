import numpy as np
import os
import casadi as ca
from aircraft import Aircraft
from simulation import Sim
from atmosphere import Atmos
from nonlinearprogramming import NLP_STG1, NLP_STG2, NLP_STG3
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
aircraft_file = sim.AIRCRAFT_FILE

# Validate that files exist.
if not os.path.isfile(aircraft_file):
    raise FileNotFoundError(f"Aircraft JSON file not found at: {aircraft_file}")

aircraft = Aircraft.from_json(aircraft_file)

print("AIRCRAFT LOADED:", aircraft.name)
print("ATMOS CONDITIONS LOADED.")

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

# STAGE 1: DESCENT. Descent flight trajectory
# defined as a NLP problem.
nlp = NLP_STG1()
w0, w, lbx, ubx, g, lbg, ubg = NLP_STG1.CONSTRAINTS_AND_BOUNDS(nlp.x,nlp.u,nlp.utf,aircraft,atmos,sim)
J = NLP_STG1.COST_FUNCTIONAL(w,aircraft,atmos,sim)

# Redefining vectors as stipulated by CASADi dictionary.
w = ca.vertcat(w)
w0 = ca.vertcat(w0)
g = ca.vertcat(*g)
lbg = ca.vertcat(*lbg)
ubg = ca.vertcat(*ubg)
lbx = ca.vertcat(*lbx)
ubx = ca.vertcat(*ubx)

# SOLVER
# Configuration of the NLP and the solver.
opts = {}
opts['ipopt.max_iter'] = 3000
opts['ipopt.tol'] = 1e-6
opts['ipopt.acceptable_tol'] = 1e-6
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
iters1 = solver.stats()['iter_count']
obj1 = solver.stats()['iterations']['obj']
time1 = np.round(solver.stats()['t_proc_total'],3)

x1 = sol['x'].full().flatten()
tF1 = x1[9*sim.N]
x1 = x1[:9*sim.N]

# Change in initial conditions. Stages chaining.
sim.w0 = x1[9*(sim.N-1):9*sim.N]

# STAGE 2: DISCHARGE. Cruise flight trajectory
# defined as a NLP problem.
nlp = NLP_STG2()
w0, w, lbx, ubx, g, lbg, ubg = NLP_STG2.CONSTRAINTS_AND_BOUNDS(nlp.x,nlp.u,nlp.utf,aircraft,atmos,sim)
J = NLP_STG2.COST_FUNCTIONAL(w,aircraft,atmos,sim)

# Redefining vectors as stipulated by CASADi dictionary.
w = ca.vertcat(w)
w0 = ca.vertcat(w0)
g = ca.vertcat(*g)
lbg = ca.vertcat(*lbg)
ubg = ca.vertcat(*ubg)
lbx = ca.vertcat(*lbx)
ubx = ca.vertcat(*ubx)

# SOLVER
# Configuration of the NLP and the solver.
opts = {}
opts['ipopt.max_iter'] = 3000
opts['ipopt.tol'] = 1e-6
opts['ipopt.acceptable_tol'] = 1e-6
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
iters2 = solver.stats()['iter_count']
obj2 = solver.stats()['iterations']['obj']
time2 = np.round(solver.stats()['t_proc_total'],3)

x2 = sol['x'].full().flatten()
tF2 = x2[9*sim.N]
x2 = x2[:9*sim.N]

# Change in initial conditions. Stages chaining.
sim.w0 = x2[9*(sim.N-1):9*sim.N]

# STAGE 3: CLIMB. Climb flight trajectory
# defined as a NLP problem.
nlp = NLP_STG3()
w0, w, lbx, ubx, g, lbg, ubg = NLP_STG3.CONSTRAINTS_AND_BOUNDS(nlp.x,nlp.u,nlp.utf,aircraft,atmos,sim)
J = NLP_STG3.COST_FUNCTIONAL(w,aircraft,atmos,sim)

# Redefining vectors as stipulated by CASADi dictionary.
w = ca.vertcat(w)
w0 = ca.vertcat(w0)
g = ca.vertcat(*g)
lbg = ca.vertcat(*lbg)
ubg = ca.vertcat(*ubg)
lbx = ca.vertcat(*lbx)
ubx = ca.vertcat(*ubx)

# SOLVER
# Configuration of the NLP and the solver.
opts = {}
opts['ipopt.max_iter'] = 3000
opts['ipopt.tol'] = 1e-6
opts['ipopt.acceptable_tol'] = 1e-6
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
iters3 = solver.stats()['iter_count']
obj3 = solver.stats()['iterations']['obj']
time3 = np.round(solver.stats()['t_proc_total'],3)

x3 = sol['x'].full().flatten()
tF3 = x3[9*sim.N]
x3 = x3[:9*sim.N]

###########################################################
##                    POSTPROCESSING                     ##
###########################################################

# PLOT GENERATION
t = np.linspace(0.0, tF1 + tF2 + tF3, 3*sim.N)

path = os.path.join(os.getcwd(), "images", "manoeuvre")
os.makedirs(path, exist_ok=True)
Plotter.GENERATE_MANOEUVRE_TRAJECTORIES(t,x1,x2,x3,aircraft,sim,path)