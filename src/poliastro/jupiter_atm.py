import math

class JovianAtmosphere:
    """
    A simple model of the Jovian atmosphere using exponential decay for pressure from https://ntrs.nasa.gov/api/citations/19790002813/downloads/19790002813.pdf
    

    """
    
    def __init__(self):
        # Reference conditions at 0 km altitude
        self.P0 = 220*101000      # Pressure at 0 km in Pa
        self.T0 = 220       # Temperature at 0 km in K
        
        # Physical constants
        self.g = 24.71         # Gravitational acceleration on Mars in m/s^2
        self.R = 189.0        # Specific gas constant for Martian CO2 atmosphere (J/(kg*K))
        self.L = 0.0013       # Lapse rate (K/m) for altitudes up to 40 km
        
        # Transition altitude between the two regions (in meters)
        self.z_transition = 40000.0  # 40 km
        
        # Precompute the exponent for the lapse rate (non-isothermal) region:
        self.exponent = self.g / (self.R * self.L)
        
        # Temperature at the transition altitude (40 km)
        self.T_transition = self.T0 - self.L * self.z_transition  # ≈ 210 - 52 = 158 K
        
        # Pressure at the transition altitude using the lapse rate formula:
        self.P_transition = self.P0 * ((self.T_transition / self.T0) ** self.exponent)
    
    def properties(self, altitude_km):

        altitude_m = altitude_km * 1000.0
        
        # Check if the altitude is within the modeled range:
        if altitude_km < -10 or altitude_km > 500:
            raise ValueError("Altitude out of modeled range (-10 km to 500 km)")
        
        if altitude_m <= self.z_transition:
            # Lower region: use linear temperature profile (lapse rate)
            T = self.T0 - self.L * altitude_m
            # Pressure using the barometric formula for a layer with a constant lapse rate
            P = self.P0 * ((T / self.T0) ** self.exponent)
        else:
            # Upper region (isothermal): Temperature remains constant at T_transition
            T = self.T_transition
            delta_z = altitude_m - self.z_transition
            # Pressure decays exponentially from the transition altitude
            P = self.P_transition * math.exp(-self.g * delta_z / (self.R * T))
        
        # Density from the ideal gas law: ρ = P / (R * T)
        density = P / (self.R * T)
        
        return {
            "altitude_m": altitude_m,
            "temperature_K": T,
            "pressure_Pa": P,
            "density_kg_m3": density
        }
    
    def __call__(self, altitude_km):
        """Allow the class instance to be called directly with an altitude in km."""
        return self.properties(altitude_km)


# # Example usage:
# if __name__ == "__main__":
#     atmosphere = MartianAtmosphere()
#     test_altitudes = [-10, 0, 10, 40, 100]  # altitudes in km
#     for alt in test_altitudes:
#         props = atmosphere.properties(alt)
#         print(f"Altitude: {alt:>4} km | Temperature: {props['temperature_K']:6.2f} K | "
#               f"Pressure: {props['pressure_Pa']:8.2f} Pa | Density: {props['density_kg_m3']:7.5f} kg/m^3")
