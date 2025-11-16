import numpy as np
import os
import casadi as ca
from aircraft import Aircraft
from atmosphere import Atmos
from nonlinearprogramming import NLP_CRUISE

# AIRCRAFT CONFIG FILE PATH
aircraft_file = "configs/FlyoxI_VI.json"

# ATMOSPHERE CONFIG FILE PATH
atmos_file = "configs/Atmos.json"

# Validate that files exist.
if not os.path.isfile(aircraft_file):
    raise FileNotFoundError(f"Aircraft JSON file not found at: {aircraft_file}")
if not os.path.isfile(atmos_file):
    raise FileNotFoundError(f"Atmosphere JSON file not found at: {atmos_file}")

aircraft = Aircraft.from_json(aircraft_file)
atmos = Atmos.from_json(atmos_file)
print("AIRCRAFT LOADED:", aircraft.name)
print("ATMOS CONDITIONS LOADED.")

class sim:
    def __init__(self):
        self.Vtp = 25
        self.w0 = 0
        self.wf = 0
 
sim = sim()

nlp = NLP_CRUISE()
w, lbx, ubx, g, lbg, ubg = NLP_CRUISE.CONSTRAINTS_AND_BOUNDS(nlp.x,nlp.u,[1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2], 0.2, 50, aircraft, atmos, sim)
J = NLP_CRUISE.COST_FUCNTIONAL(w,0.2,50,aircraft,atmos)

nlp = {"x": ca.vertcat(w), "f": J, "g": ca.vertcat(*g)}

solver = ca.nlpsol("solver", "ipopt", nlp)

sol = solver(
    x0 = np.zeros(50*9),
    lbx = lbx,
    ubx = ubx,
    lbg = lbg,
    ubg = ubg
)

