import numpy as np
import json

class Atmos:
    def __init__(
            self,
            GRAVITY_ACC: float,
            SL_PRESS: float,
            SL_TEMP: float,
            SL_RHO: float,
            ATMOS_GRADIENT: float,
            AIR_CONSTANT: float,
        ):
            # PARAMETERS
            self.g = GRAVITY_ACC
            self.P0 = SL_PRESS
            self.T0 = SL_TEMP
            self.rho0 = SL_RHO
            self.L = ATMOS_GRADIENT
            self.R = AIR_CONSTANT
            self.REarth = 6371000.0
    
    @classmethod
    def from_json(cls, filepath: str) -> "Atmos":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "GRAVITY_ACC", "SL_PRESS", "SL_TEMP", 
            "SL_RHO", "ATMOS_GRADIENT", "AIR_CONSTANT"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in ATMOS JSON: {missing}")

        return cls(**data)
    
    def ISA_RHO(self, h: float):
            # Geopotential altitude correction.
            h = self.REarth * h / (self.REarth + h)

            # Computation of air's density.
            T = self.T0 + 273.15 + self.L * h
            P = self.P0 * (T / (273.15 + self.T0)) ** (-self.g / (self.R * self.L))
            rho = P / (self.R * T)

            return rho