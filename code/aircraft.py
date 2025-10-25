import numpy as np
import json

class Aircraft:
    def __init__(
        self,
        name: str,
        BEM: float,
        FM: float,
        PM: float,
        SFC: float,
        AR: float,
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
        CRUISE_SPECS=None,
        CLIMB_SPECS=None,
        DESCENT_SPECS=None
    ):
        # CORE PARAMETERS
        self.name = name
        self.BEM = BEM
        self.FM = FM
        self.PM = PM
        self.SFC = SFC
        self.AR = AR
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

        # Extracting aerodynamic force and moments coefficients for 
        # the differents flap configurations and for the elevator.
        self.CL_CD_F0 = self.LIFT_DRAG_COEFFS[0]
        self.CL_CD_F15 = self.LIFT_DRAG_COEFFS[1]
        self.CL_CD_F40 = self.LIFT_DRAG_COEFFS[2]
        self.CLde = self.ELEVATOR_COEFFS[0]
        self.Cmde = self.ELEVATOR_COEFFS[1]

        # Optional SPECS. Flight phases constraints.
        self.CRUISE_SPECS = np.array(CRUISE_SPECS) if CRUISE_SPECS is not None else None
        self.CLIMB_SPECS = np.array(CLIMB_SPECS) if CLIMB_SPECS is not None else None
        self.DESCENT_SPECS = np.array(DESCENT_SPECS) if DESCENT_SPECS is not None else None
    
    @classmethod
    def from_json(cls, filepath: str) -> "Aircraft":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "name","BEM","FM","PM","SFC","AR","b","c","e",
            "CGx","CGz","CPwx","CPwz","CTx","CTz","Dp","nENG",
            "EPS0","Iyy","LIFT_DRAG_COEFFS","MOMENT_COEFFS",
            "ELEVATOR_COEFFS","THRUST_COEFFS","EFFICIENCY_COEFFS"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in AIRCRAFT JSON: {missing}")

        return cls(**data)
