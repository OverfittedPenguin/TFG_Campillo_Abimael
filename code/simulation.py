import json
import numpy as np

aircraft = ()

class Sim:
    def __init__(
            self,
            TIME_INCREMENT: float,
            END_TIME: float,
            WIND_SPEED: list,
            INITIAL_GUESS: list,
            ADDITIONAL_CONSTRAINTS: list,
            END_TIME_CONSTRAINTS: list,
            AIRCRAFT_FILE: str
        ):
            # PARAMETERS
            self.dT = TIME_INCREMENT
            self.tF = END_TIME
            self.Wind = WIND_SPEED
            self.Nodes = int(self.tF / self.dT)
            self.w0 = INITIAL_GUESS

            self.CONSTRAINTS = np.array(ADDITIONAL_CONSTRAINTS)
            self.Vtp = self.CONSTRAINTS[0]

            self.wf = END_TIME_CONSTRAINTS

            self.Aircraft_file = AIRCRAFT_FILE
    
    @classmethod
    def from_json(cls, filepath: str) -> "Sim":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "TIME_INCREMENT", "END_TIME", "WIND_SPEED", "INITIAL_GUESS", 
            "END_TIME_CONSTRAINTS", "ADDITIONAL_CONSTRAINTS", "AIRCRAFT_FILE"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in SIMULATION JSON: {missing}")

        return cls(**data)