import json
import numpy as np

aircraft = ()

class Sim:
    def __init__(
            self,
            DT: float,
            INITIAL_STATE: list,
            END_STATE: list,
            WIND_SPEED: list,
            TARGET_POINT: list,
            MISSION_BOUNDS: list,
            STG1_WEIGHTS:list,
            STG2_WEIGHTS:list,
            STG3_WEIGHTS:list,
            AIRCRAFT_FILE: str
        ):
            # PARAMETERS  
            self.dT = DT  
            self.N = int(1 / self.dT)

            self.x0_USER = np.array(INITIAL_STATE)
            self.V0 = float(self.x0_USER[0])
            self.theta0 = float(self.x0_USER[1])
            self.t0 = float(self.x0_USER[2])
            self.x0 = np.zeros(8)
            self.w0 = 0.0

            self.wf = END_STATE

            self.wind = np.array(WIND_SPEED)

            self.target = np.array(TARGET_POINT)
            self.R_entry = float(self.target[0])
            self.R_exit = float(self.target[1])
            self.f_tp = float(self.target[2])
            self.Vtp = float(self.target[3])

            self.bounds = np.array(MISSION_BOUNDS)
            self.lb = self.bounds[0]
            self.ub = self.bounds[1]
            
            self.stg1_wg = np.array(STG1_WEIGHTS)
            self.stg2_wg = np.array(STG2_WEIGHTS)
            self.stg3_wg = np.array(STG3_WEIGHTS)

            self.Aircraft_file = AIRCRAFT_FILE

            # INITIAL STATE COMPUTATIONS
            # Body velocities. Aerodynamic velocity.
            u, w = self.V0 / np.cos(self.theta0), self.V0 / np.sin(self.theta0)
            ua, wa = u - self.wind[0], w - self.wind[1]
            V = np.sqrt(ua**2 + wa**2)
            self.x0[0] = V
            self.x0[1] = u
            self.x0[2] = w

            # Pitch rate. Always zero.
            self.x0[3] = 0.0

            # Pitch. User input.
            self.x0[4] = self.theta0

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
            "DT", "INITIAL_STATE", "END_STATE", "WIND_SPEED",
            "TARGET_POINT", "MISSION_BOUNDS", "STG1_WEIGHTS",
            "STG2_WEIGHTS", "STG3_WEIGHTS", "AIRCRAFT_FILE"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in SIMULATION JSON: {missing}")

        return cls(**data)