import json
import numpy as np

aircraft = ()

class Sim:
    def __init__(
            self,
            END_TIME: float,
            N: list,
            INITIAL_STATE: list,
            END_TIME_CONSTRAINTS: list,
            ADDITIONAL_CONSTRAINTS: list,
            WIND_SPEED: list,
            CRUISE_WEIGHTS:list,
            AIRCRAFT_FILE: str
        ):
            # PARAMETERS
            self.tF = END_TIME
            self.N = np.array(N)
            self.dT = self.tF / N
    
            self.x0_USER = np.array(INITIAL_STATE)
            self.x0 = np.zeros(11)
            self.w0 = 0.0
            self.CONSTRAINTS = np.array(ADDITIONAL_CONSTRAINTS)
            self.Vtp = self.CONSTRAINTS[0]
            self.wf = END_TIME_CONSTRAINTS

            self.wind = np.array(WIND_SPEED)
            self.cruise_wg = np.array(CRUISE_WEIGHTS)

            self.Aircraft_file = AIRCRAFT_FILE

            # INITIAL STATE COMPUTATIONS
            # Re-arranging of initial altitude AGL.
            self.x0[6] = self.x0_USER[2] 

            # Initial aerodynamic velocity.
            ua, wa = self.x0_USER[0] - self.wind[0], self.x0_USER[1] - self.wind[1]
            self.x0[2] = self.x0_USER[1]
            self.x0[1] = self.x0_USER[0]
            self.x0[0] = np.sqrt(ua**2 + wa**2)

            # Initial position. Always zero.
            self.x0[5] = 0.0

            # Initial pitch rate. Always zero.
            self.x0[3] = 0.0

            # Initial pitch. Equal to initial AoA.
            self.x0[4] = np.abs(np.arctan2(wa,ua))

    
    @classmethod
    def from_json(cls, filepath: str) -> "Sim":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "END_TIME", "N", "INITIAL_STATE", 
            "END_TIME_CONSTRAINTS", "ADDITIONAL_CONSTRAINTS",
            "WIND_SPEED", "CRUISE_WEIGHTS", "AIRCRAFT_FILE"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in SIMULATION JSON: {missing}")

        return cls(**data)