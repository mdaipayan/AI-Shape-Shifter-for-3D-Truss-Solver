import pandas as pd
import streamlit as st

@st.cache_data
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
        ["ISA 20x20x3", 1.12, 0.38, 0.9],
    ["ISA 25x25x3", 1.41, 0.48, 1.1],
    ["ISA 30x30x3", 1.73, 0.58, 1.4],
    ["ISA 35x35x5", 3.28, 0.67, 2.6],
    ["ISA 40x40x3", 2.32, 0.78, 1.8],
    ["ISA 40x40x4", 3.08, 0.77, 2.4],
    ["ISA 40x40x5", 3.78, 0.76, 3.0],
    ["ISA 40x40x6", 4.47, 0.76, 3.5],
    ["ISA 45x45x3", 2.62, 0.88, 2.1],
    ["ISA 45x45x4", 3.48, 0.87, 2.7],
    ["ISA 45x45x5", 4.27, 0.87, 3.4],
    ["ISA 45x45x6", 5.07, 0.86, 4.0],
    ["ISA 50x50x3", 2.92, 0.98, 2.3],
    ["ISA 50x50x4", 3.88, 0.98, 3.0],
    ["ISA 50x50x5", 4.79, 0.98, 3.8],
    ["ISA 50x50x6", 5.68, 0.97, 4.5],
    ["ISA 55x55x5", 5.27, 1.08, 4.1],
    ["ISA 55x55x6", 6.26, 1.07, 4.9],
    ["ISA 60x60x5", 5.75, 1.18, 4.5],
    ["ISA 60x60x6", 6.84, 1.16, 5.4],
    ["ISA 60x60x8", 8.96, 1.15, 7.0],
    ["ISA 65x65x5", 6.30, 1.26, 4.9],
    ["ISA 65x65x6", 7.44, 1.26, 5.8],
    ["ISA 65x65x8", 9.76, 1.25, 7.7],
    ["ISA 70x70x5", 6.80, 1.36, 5.3],
    ["ISA 70x70x6", 8.06, 1.36, 6.3],
    ["ISA 75x75x5", 7.30, 1.47, 5.7],
    ["ISA 75x75x6", 8.66, 1.46, 6.8],
    ["ISA 75x75x8", 11.38, 1.45, 8.9],
    ["ISA 75x75x10", 14.02, 1.43, 11.0],
    ["ISA 80x80x6", 9.29, 1.56, 7.3],
    ["ISA 80x80x8", 12.21, 1.55, 9.6],
    ["ISA 90x90x6", 10.47, 1.76, 8.2],
    ["ISA 90x90x8", 13.79, 1.75, 10.8],
    ["ISA 90x90x10", 17.03, 1.74, 13.4],
    ["ISA 100x100x6", 11.67, 1.96, 9.2],
    ["ISA 100x100x8", 15.39, 1.95, 12.1],
    ["ISA 100x100x10", 19.03, 1.94, 14.9],
    ["ISA 100x100x12", 22.59, 1.93, 17.7],
    ["ISA 110x110x8", 17.02, 2.15, 13.4],
    ["ISA 110x110x10", 21.06, 2.14, 16.5],
    ["ISA 110x110x12", 25.02, 2.13, 19.6],
    ["ISA 130x130x8", 20.22, 2.55, 15.9],
    ["ISA 130x130x10", 25.13, 2.54, 19.7],
    ["ISA 130x130x12", 29.82, 2.53, 23.4],
    ["ISA 150x150x10", 29.03, 2.94, 22.8],
    ["ISA 150x150x12", 34.59, 2.93, 27.2],
    ["ISA 150x150x15", 42.78, 2.91, 33.6],
    ["ISA 200x200x12", 46.61, 3.93, 36.6],
    ["ISA 200x200x15", 57.80, 3.91, 45.4],
    ["ISA 200x200x25", 93.80, 3.85, 73.6]
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
