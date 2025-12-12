import json
import numpy as np

aircraft = ()

class Sim:
    def __init__(
            self,
            N: float,
            INITIAL_STATE: list,
            WIND_SPEED: list,
            TARGET_POINT: list,
            MISSION_BOUNDS: list,
            STG1_WEIGHTS:list,
            STG2_WEIGHTS:list,
            STG3_WEIGHTS:list,
            AIRCRAFT_FILE: str
        ):
            # PARAMETERS  
            self.N = N
            self.dT = 1 / (N-1)
            
            self.x0_USER = np.array(INITIAL_STATE)
            self.V0 = float(self.x0_USER[0])
            self.th0 = float(self.x0_USER[1])
            self.t0 = self.x0_USER[2:5]
            self.x0 = np.zeros(8)
            self.w0 = 0.0

            self.wind = np.array(WIND_SPEED)

            self.TARGET = np.array(TARGET_POINT)
            self.R_entry = float(self.TARGET[0])
            self.R_exit = float(self.TARGET[1])
            self.f_tp = float(self.TARGET[2])
            self.Vtp = float(self.TARGET[3])
            self.td = float(self.TARGET[4])

            self.BOUNDS = np.array(MISSION_BOUNDS)
            self.lb = self.BOUNDS[0]
            self.ub = self.BOUNDS[1]
            
            self.STG1_wg = np.array(STG1_WEIGHTS)
            self.STG2_wg = np.array(STG2_WEIGHTS)
            self.STG3_wg = np.array(STG3_WEIGHTS)

            self.AIRCRAFT_FILE = AIRCRAFT_FILE

            # INITIAL STATE COMPUTATIONS
            # Body velocities. Aerodynamic velocity.
            u, w = self.V0 / np.cos(self.th0), self.V0 / np.sin(self.th0)
            ua, wa = u - self.wind[0], w - self.wind[1]
            V = np.sqrt(ua**2 + wa**2)
            self.x0[0] = V
            self.x0[1] = u
            self.x0[2] = w

            # Pitch rate. Always zero.
            self.x0[3] = 0.0

            # Pitch. User input.
            self.x0[4] = self.th0

            # Initial distance and altitude. Zero and target point altitude (altitude ub, cruise altitude).
            self.x0[5] = 0.0
            self.x0[6] = self.ub[3]
    
    @classmethod
    def from_json(cls, filepath: str) -> "Sim":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "N", "INITIAL_STATE", "WIND_SPEED", "TARGET_POINT",
            "MISSION_BOUNDS", "STG1_WEIGHTS", "STG2_WEIGHTS",
            "STG3_WEIGHTS", "AIRCRAFT_FILE"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in SIMULATION JSON: {missing}")

        return cls(**data)