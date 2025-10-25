import numpy as np
import json

class Atmos:
    def __init__(
            self,
            GRAVITY_ACC: float,
            SL_PRESS: float,
            SL_TEMP: float,
            SL_RHO: float,
            GRADIENT: float,
            AIR_CONSTANT: float,
            WIND_SPEED: float,
        ):
            # PARAMETERS
            self.g = GRAVITY_ACC
            self.P0 = SL_PRESS
            self.T0 = SL_TEMP
            self.rho0 = SL_RHO
            self.L = GRADIENT
            self.R = AIR_CONSTANT
            self.Wind = WIND_SPEED
            self.REarth = 6371000.0
    
    @classmethod
    def from_json(cls, filepath: str) -> "Atmos":
        # Validation.
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # ESSENTIAL KEYS
        essential_keys = [
            "GRAVITY_ACC", "SL_PRESS", "SL_TEMP", "SL_RHO", "GRADIENT",
            "AIR_CONSTANT", "WIND_SPEED"
        ]
        missing = [k for k in essential_keys if k not in data]
        if missing:
            raise KeyError(f"Missing essential keys in ATMOS JSON: {missing}")

        return cls(**data)
    
    def ISA_RHO(self, h: float):
            
            # Geopotential altitude correction.
            h = self.REarth * h / (self.REarth + h)

            # Computation of density and its correction factor.
            T = self.T0 + 273.15 + self.L * h
            P = self.P0 * (T / (273.15 + self.T0)) ** (-self.g / (self.R * self.L))
            rho = P / (self.R * T)
            if h <= 914.0:
                # Not correction applied below 914m (3000ft)
                sigma = 1.0
            else:
                sigma = rho / self.rho0
                
            return rho, sigma