import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=1)
def get_isa_catalog():
    """
    Returns a Pandas DataFrame containing standard Indian Standard Equal Angles (ISA)
    properties extracted from IS 800 / SP 6(1).
    
    Units in the catalog:
    - Area: cm^2
    - r_min (minimum radius of gyration, r_vv): cm
    - Weight: kg/m
    """
    
    catalog_data = [
        # ["Designation", Area(cm2), r_min(cm), Weight(kg/m)]
        ["ISA 40x40x4", 3.08, 0.77, 2.4],
        ["ISA 40x40x5", 3.78, 0.76, 3.0],
        ["ISA 45x45x5", 4.27, 0.87, 3.4],
        ["ISA 50x50x5", 4.79, 0.98, 3.8],
        ["ISA 50x50x6", 5.68, 0.97, 4.5],
        ["ISA 60x60x6", 6.84, 1.16, 5.4],
        ["ISA 65x65x6", 7.44, 1.26, 5.8],
        ["ISA 75x75x6", 8.66, 1.46, 6.8],
        ["ISA 75x75x8", 11.38, 1.45, 8.9],
        ["ISA 90x90x6", 10.47, 1.76, 8.2],
        ["ISA 90x90x8", 13.79, 1.75, 10.8],
        ["ISA 100x100x8", 15.39, 1.95, 12.1],
        ["ISA 100x100x10", 19.03, 1.94, 14.9],
        ["ISA 110x110x10", 21.06, 2.14, 16.5],
        ["ISA 130x130x10", 25.13, 2.54, 19.7],
        ["ISA 130x130x12", 29.82, 2.53, 23.4],
        ["ISA 150x150x10", 29.03, 2.94, 22.8],
        ["ISA 150x150x15", 42.78, 2.93, 33.6],
        ["ISA 200x200x15", 57.78, 3.93, 45.4],
        ["ISA 200x200x25", 93.80, 3.89, 73.6]
    ]
    
    df = pd.DataFrame(catalog_data, columns=["Designation", "Area_cm2", "r_min_cm", "Weight_kg_m"])
    
    # Pre-calculate base SI units (m^2 and meters) so the solver doesn't have to do it 10,000 times
    df["Area_m2"] = df["Area_cm2"] / 10000.0
    df["r_min_m"] = df["r_min_cm"] / 100.0
    
    return df

# If you run this file directly, it will just print the table to test it
if __name__ == "__main__":
    catalog = get_isa_catalog()
    print("Indian Standard Angle (ISA) Catalog Loaded:")
    print("-" * 60)
    print(catalog.head(20))
