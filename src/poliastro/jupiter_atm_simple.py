import math

class JovianAtmosphereSimple:
    """
    A simple model of the Jovian atmosphere using exponential decay for pressure from https://ntrs.nasa.gov/api/citations/19790002813/downloads/19790002813.pdf
    

    """
    
    def __init__(self):
        # Reference conditions at 0 km altitude
        self.P0 = 220*101000      # Pressure at 0 km in Pa
        self.T0 = 220       # Temperature at 0 km in K
        
    
    def properties(self, altitude_km):
        # Check if the altitude is within the modeled range:
        if altitude_km.any() < -10 or altitude_km.any() > 500:
            raise ValueError("Altitude out of modeled range (-10 km to 500 km)")
        
        pressure = self.P0 * math.exp(-altitude_km / 27)
        return{
            # "altitude_m": altitude_m,
            # "temperature_K": T,
            "pressure_Pa": pressure
            # "density_kg_m3": density
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
