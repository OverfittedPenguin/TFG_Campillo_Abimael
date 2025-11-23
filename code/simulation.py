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
            AIRCRAFT_FILE: str
        ):
            # PARAMETERS
            self.tF = END_TIME
            self.N = np.array(N)
            self.dT = self.tF / N
    
            self.w0 = INITIAL_STATE
            self.CONSTRAINTS = np.array(ADDITIONAL_CONSTRAINTS)
            self.Vtp = self.CONSTRAINTS[0]
            self.wf = END_TIME_CONSTRAINTS

            self.wind = WIND_SPEED

            self.Aircraft_file = AIRCRAFT_FILE
    
    @classmethod
    def from_json(cls, filepath: str) -> "Sim":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "END_TIME", "N", "INITIAL_STATE", 
            "END_TIME_CONSTRAINTS", "ADDITIONAL_CONSTRAINTS",
            "WIND_SPEED", "AIRCRAFT_FILE"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in SIMULATION JSON: {missing}")

        return cls(**data)