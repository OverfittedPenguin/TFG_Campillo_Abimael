import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
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

nlp = NLP_CRUISE()
w = NLP_CRUISE.STATES_CONTROL_VECTOR(nlp.x,nlp.u,2)
print(w)
w1 = w[:9]
w2 = w[9:18]
print(w1, w2)
f1 = NLP_CRUISE.DYNAMIC_EQUATIONS(w1,aircraft,atmos)
f2 = NLP_CRUISE.DYNAMIC_EQUATIONS(w2,aircraft,atmos)
print(f1,f2)
g, lbg, ubg = NLP_CRUISE.DYNAMIC_CONSTRAINTS(w1[:7],w2[:7],f1[:7],f2[:7],0.2)
print(g, lbg, ubg)
print(len(g))
print(len(lbg))
print(len(ubg))