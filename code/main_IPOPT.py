import numpy as np
import os
import casadi as ca
from aircraft import Aircraft
from simulation import Sim
from atmosphere import Atmos
from nonlinearprogramming import NLP_CRUISE
from plotterfunction import Plotter
import matplotlib.pyplot as plt

###########################################################
##                  USER CONFIGURATION                   ##
###########################################################

# SIMULATION CONFIG FILE PATH
simulation_file = "configs/Simulation.json"

# ATMOSPHERE CONFIG FILE PATH
atmos_file = "configs/Atmos.json"

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

# Initial state vector
sim.w0 = ca.vertcat(x0, [1.0, 0.0])

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

# States and controls storage vectors.
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
    # For each variable, the current value is retrived and 
    # appended into the respecive storage array.

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

# Arrays conversion.
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
x5 = np.array(x5)
x6 = np.array(x6)
x7 = np.array(x7)
u1 = np.array(u1)
u2 = np.array(u2)

# FORCES RECOSNTRUCTION
Fxb_A = []
Fzb_A = []
Fxb_T = []
Fxb_W = []
Fzb_W = []

for k in range(len(x1)):
    # Computation of atmosphere parameters,
    # aerodynamic velocity and dynamic pressure.
    rho = atmos.ISA_RHO(-x6[k])
    ua, wa = x1[k] - sim.wind[0], x2[k] - sim.wind[1]
    alpha = np.arctan2(wa, ua)
    V = np.sqrt(ua**2 + wa**2)
    q_bar = 0.5*rho*V**2

    # AERODYNAMIC COEFFICIENTS AND FORCES
    CL = aircraft.CL_0 + aircraft.CL_alpha * alpha + aircraft.CL_de * u2[k]
    CD = aircraft.CD_0 + aircraft.K * CL**2
    Rsb = np.array([
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])
    Cb_aero = Rsb.T @ np.array([-CD, -CL])
    Fb_aero = q_bar * aircraft.S * Cb_aero
    Fxb_A.append(Fb_aero[0])
    Fzb_A.append(Fb_aero[1])

    # PROPULSIVE FORCE
    T_max,M_T_max = aircraft.PROPULSIVE_FORCES_MOMENTS(V,aircraft.RPM,rho,alpha)
    Fxb_T.append(u1[k] * T_max)

    # WEIGHT
    Rhb = np.array([
        [np.cos(x4[k]), np.sin(x4[k])],
        [-np.sin(x4[k]), np.cos(x4[k])]
    ])
    Fb_grav = Rhb.T @ np.array([0.0, x7[k] * atmos.g])
    Fxb_W.append(Fb_grav[0])
    Fzb_W.append(Fb_grav[1])

# Computation of negative Fxb.
Fxb = [a + b for a,b in zip(Fxb_A,Fxb_W)]

# AoA RECONSTRUCTION
alpha = np.arctan2(x2 - sim.wind[1], x1 - sim.wind[0])

# PLOTS
path = "/home/abimael_campillo/Desktop/TFG_Campillo_Abimael/code/images/benchmark"
Plotter.GENERATE_PLOT(t,np.column_stack((x1, x2)),["u","w"],["Time [s]", "Velocity [m/s]","Body velocities through time","VEL.png"],path)
Plotter.GENERATE_PLOT(t,np.column_stack((x4, alpha)),[r"$\theta$",r"$\alpha$"],["Time [s]", "Angles [rad]","Pitch and AoA through time","ANGLES.png"],path)
Plotter.GENERATE_PLOT(t,x7,"Mass",["Time [s]", "Mass [kg]","Aircraft's mass through time","MASS.png"],path) 
Plotter.GENERATE_PLOT(t,np.column_stack((Fxb, Fxb_T)),["Aerodynamic + Weight", "Thrust"],["Time [s]", "Forces [N]","Body x-axis forces through time","FORCES_Xb.png"],path)
Plotter.GENERATE_PLOT(t,np.column_stack((Fzb_A, Fzb_W)),["Aerodynamic", "Weight"],["Time [s]", "Forces [N]","Body z-axis forces through time","FORCES_Zb.png"],path)  
Plotter.GENERATE_PLOT(t,u1,r"$\delta_T$",["Time [s]", "TPS [-]","Throttle position through time","CONTROL_dT.png"],path)
Plotter.GENERATE_PLOT(t,u2,r"$\delta_e$",["Time [s]", "Elevator [rad]","Elevator deflection through time","CONTROL_de.png"],path)
Plotter.GENERATE_PLOT(x5,-x6,"Trajectory",["Horizontal distance [m]", "Altitude AGL [m]","Aircraft's trajectory","TRAJECTORY.png"],path)
Plotter.GENERATE_PLOT(np.linspace(1,iters+1,len(obj)),np.array(obj),"Objective cost",["Iterations [-]", "Cost objective [-]", "Cost evolution. Total computation time: " f"{time} s", "COST.png"], path)
