import numpy as np
import casadi as ca
import json

class Aircraft:
    def __init__(
        self,
        NAME: str,
        BEM: float,
        MTOM: float,
        FM: float,
        PM: float,
        SFC: float,
        AR: float,
        S: float,
        b: float,
        c: float,
        e: float,
        CGx: float,
        CGz: float,
        CPwx: float,
        CPwz: float,
        CTx: float,
        CTz: float,
        Dp: float,
        nENG: float,
        EPS0: float,
        Iyy: float,
        LIFT_DRAG_COEFFS: list,
        MOMENT_COEFFS: list,
        ELEVATOR_COEFFS: list,
        THRUST_COEFFS: list,
        EFFICIENCY_COEFFS: list,
        OPERATIONAL_LIMITS: list,
        MISSION_SPECS: list,
    ):
        # CORE PARAMETERS
        self.name = NAME
        self.BEM = BEM
        self.MTOM = MTOM
        self.FM = FM
        self.PM = PM
        self.SFC = SFC
        self.AR = AR
        self.S = S
        self.b = b
        self.c = c
        self.e = e
        self.K = 1 / (np.pi * self.AR * self.e)

        self.CGx = CGx
        self.CGz = CGz
        self.CPwx = CPwx
        self.CPwz = CPwz
        self.CTx = CTx
        self.CTz = CTz
        self.DeltaX_WG = self.CPwx - self.CGx
        self.DeltaZ_WG = self.CPwz - self.CGz
        self.DeltaX_T = self.CPwz - self.CGx
        self.DeltaZ_T = self.CPwz - self.CGz

        self.Dp = Dp
        self.nENG = nENG
        self.EPS0 = EPS0
        self.Iyy = Iyy

        # Conversion to arrays.
        self.LIFT_DRAG_COEFFS = np.array(LIFT_DRAG_COEFFS)
        self.MOMENT_COEFFS = np.array(MOMENT_COEFFS)
        self.ELEVATOR_COEFFS = np.array(ELEVATOR_COEFFS)
        self.THRUST_COEFFS = np.array(THRUST_COEFFS)
        self.EFFICIENCY_COEFFS = np.array(EFFICIENCY_COEFFS)
        self.MISSION_SPECS = np.array(MISSION_SPECS)
        self.BOUNDS = np.array(OPERATIONAL_LIMITS)
        self.lb_USER = self.BOUNDS[0]
        self.ub_USER = self.BOUNDS[1]

        # BOUNDS
        self.lb = np.zeros(9)
        self.ub = np.zeros(9)

        # Computation of maximum and minimum body velocities.
        umax, wmax = self.ub_USER[0] * np.cos(self.ub_USER[1]), self.ub_USER[0] * np.sin(self.ub_USER[1])
        umin, wmin = 0.0, self.lb_USER[0] * np.sin(self.lb_USER[1])

        self.lb[0] = umin
        self.lb[1] = wmin
        self.ub[0] = umax
        self.ub[1] = wmax

        # Pitch rate and pitch bounds.
        self.lb[2] = 0.0
        self.lb[3] = self.lb_USER[1]
        self.ub[2] = 0.0
        self.ub[3] = self.ub_USER[1]

        # Distance and altitude bounds.
        self.lb[4] = self.lb_USER[2]
        self.lb[5] = self.lb_USER[3]
        self.ub[4] = self.ub_USER[2]
        self.ub[5] = self.ub_USER[3]

        # Mass bounds.
        self.lb[6] = self.BEM
        self.ub[6] = self.MTOM

        # Control bounds.
        self.lb[7] = self.lb_USER[4]
        self.lb[8] = self.lb_USER[5]
        self.ub[7] = self.ub_USER[4]
        self.ub[8] = self.ub_USER[5]

        # Extracting aerodynamic force and moments coefficients for 
        # different flap configurations and for elevator contribution.
        self.CL_CD_F0 = self.LIFT_DRAG_COEFFS[0]
        self.CL_CD_F15 = self.LIFT_DRAG_COEFFS[1]
        self.CL_CD_F40 = self.LIFT_DRAG_COEFFS[2]
        self.Cm_0 = self.MOMENT_COEFFS[0]
        self.Cm_alpha = self.MOMENT_COEFFS[1]
        self.CL_de = self.ELEVATOR_COEFFS[0]
        self.Cm_de = self.ELEVATOR_COEFFS[1]

        # Extracting operational limits and mission specs.
        self.FLAPS = self.MISSION_SPECS[0]
        self.RPM = self.MISSION_SPECS[1]

        # Aerodynamic coefficients that are going to be used.
        if self.FLAPS == 0.0:
            self.CL_0 = self.CL_CD_F0[0]
            self.CL_alpha = self.CL_CD_F0[1]
            self.CD_0 = self.CL_CD_F0[2]
        elif self.FLAPS == 15.0:
            self.CL_0 = self.CL_CD_F15[0]
            self.CL_alpha = self.CL_CD_F15[1]
            self.CD_0 = self.CL_CD_F15[2]
        elif self.FLAPS == 40.0:
            self.CL_0 = self.CL_CD_F40[0]
            self.CL_alpha = self.CL_CD_F40[1]
            self.CD_0 = self.CL_CD_F40[2]
        else:
            raise ValueError(f"Flaps set doesn't met expected values F0, F15 or F40.")
    
    @classmethod
    def from_json(cls, filepath: str) -> "Aircraft":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "NAME","BEM","MTOM","FM","PM","SFC","AR","b","c","e",
            "CGx","CGz","CPwx","CPwz","CTx","CTz","Dp","nENG",
            "EPS0","Iyy","LIFT_DRAG_COEFFS","MOMENT_COEFFS",
            "ELEVATOR_COEFFS","THRUST_COEFFS","EFFICIENCY_COEFFS", 
            "OPERATIONAL_LIMITS", "MISSION_SPECS"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in AIRCRAFT JSON: {missing}")

        return cls(**data)
    
    def PROPULSIVE_FORCES_MOMENTS(self, V, n: float, rho, AOA):
        # Advance ratio computation using RPS.
        n_RPS = n / 60.0
        J = V / (n_RPS * self.Dp)
        eps = np.deg2rad(self.EPS0) - AOA
        
        # Ct and efficiency for computed J.
        Ct = self.THRUST_COEFFS[0] + self.THRUST_COEFFS[1]*J + self.THRUST_COEFFS[2]*J**2 + self.THRUST_COEFFS[3]*J**3
        eta = self.EFFICIENCY_COEFFS[0] + self.EFFICIENCY_COEFFS[1]*ca.log(J)

        # Thrust and longitudinal torque.
        T = self.nENG * eta * rho * n_RPS**2 * self.Dp**4 * Ct
        M_T = -T * self.DeltaX_T * ca.sin(eps) - T * self.DeltaZ_T * ca.cos(eps)
  
        return T,M_T
