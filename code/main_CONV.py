import numpy as np
import os
import casadi as ca
from aircraft import Aircraft
from simulation import Sim
from atmosphere import Atmos
from nonlinearprogramming import NLP_CRUISE
from plotterfunction import Plotter


###########################################################
##                  USER CONFIGURATION                   ##
###########################################################

# SIMULATION CONFIG FILE PATH
simulation_file = "configs/Simulation.json"

# ATMOSPHERE CONFIG FILE PATH
atmos_file = "configs/Atmos.json"

# Evaluation nodes. Convergence study.
N = [25,50,75,100,125,150]

###########################################################
##              PREPROCESS OF FILES AND VARS             ##
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

# Initial rho.
sim.x0[8] = atmos.ISA_RHO(-sim.x0[6])

# Computation of trim conditions for controls at initial state.
x0 = sim.x0[1:8]
dt0, de0 = NLP_CRUISE.TRIM_CONTROLS(x0,aircraft,atmos,sim)
dt0, de0 = np.round(dt0,3), np.round(de0,3)
print("INITIAL CONTROLS: dt = " f"{dt0} de = {de0}")

# Initial state vector
sim.w0 = np.concatenate((sim.x0[1:8], [0.95, -0.069]))

###########################################################
##          PROBLEM DEFINITION AND SOLUTION              ##
###########################################################

# PREPROCESS

# CRUISE SUBPROBLEM DEFINITION. Cruise flight trajectory
# defined as a NLP problem.
for j in range(len(N)):

    sim.N = N[j]
    sim.dT = sim.tF / sim.N

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

    # Storage.
    t_total = []
    t_total.append(t)

    x1_total = []
    x2_total = []
    x4_total = []
    x5_total = []
    x6_total = []
    u1_total = []
    u2_total = []

    AoA_total = []
    iters_total = []
    iters_total.append(iters)
    obj_total = []
    obj_total.append(obj)
    eval_time = []
    eval_time.append(time)

    # States and controls storage vectors.
    x1 = []
    x2 = []
    x4 = []
    x5 = []
    x6 = []
    u1 = []
    u2 = []

    for k in range(sim.N):
        # For each variable, the current value is retrived and 
        # appended into the respecive storage array.

        idx = 9*k
        x1.append(x[idx])
        x2.append(x[idx + 1])
        x4.append(x[idx + 3])
        x5.append(x[idx + 4])
        x6.append(x[idx + 5])
        u1.append(x[idx + 7])
        u2.append(x[idx + 8])

    # Append of values for a number of nodes.
    x1_total.append(x1)
    x2_total.append(x2)
    x4_total.append(x4)
    x5_total.append(x5)
    x6_total.append(x6)
    u1_total.append(u1)
    u2_total.append(u2)

    # AoA RECONSTRUCTION
    alpha = np.arctan2(x2 - sim.wind[1], x1 - sim.wind[0])
    AoA_total.append(alpha)

# Conversion to arrays.
t_total = np.array(t_total)
iters_total = np.array(iters_total)
obj_total = np.array(obj_total)
eval_time = np.array(eval_time)
x1_total = np.array(x1_total)
x2_total = np.array(x2_total)
x4_total = np.array(x4_total)
x5_total = np.array(x5_total)
x6_total = np.array(x6_total)
u1_total = np.array(u1_total)
u2_total = np.array(u2_total)

# PLOTS

