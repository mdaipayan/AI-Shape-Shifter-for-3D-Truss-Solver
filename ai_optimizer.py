import numpy as np
from scipy.optimize import differential_evolution
import copy
from is_catalog import get_isa_catalog

class TrussOptimizer:
    def __init__(self, base_truss, member_groups=None, yield_stress=250e6, max_deflection=0.05):
        """
        Discrete AI Optimizer using IS 800 Catalog and Member Grouping.
        member_groups: List of lists, e.g., [[1, 2, 3], [4, 5, 6]] where numbers are Member IDs.
        """
        # Make a deep copy so the AI doesn't corrupt the UI model during thousands of iterations
        self.ts = copy.deepcopy(base_truss) 
        self.catalog = get_isa_catalog()
        self.yield_stress = yield_stress
        self.max_deflection = max_deflection
        
        # Setup Grouping (Symmetry)
        if member_groups is None:
            # If no grouping is provided, every member is optimized independently
            self.member_groups = [[m.id] for m in self.ts.members]
        else:
            self.member_groups = member_groups
            
        self.num_groups = len(self.member_groups)

    def objective_function(self, group_indices):
        """
        The Evaluation Loop for DISCRETE catalog sections.
        group_indices: Array of integer indices pointing to rows in the IS Catalog.
        """
        weight = 0.0
        
        # 1. Apply catalog properties to members based on their group
        for group_idx, member_ids in enumerate(self.member_groups):
            # SciPy might pass floats internally, so we ensure it's a valid integer index
            cat_idx = int(round(group_indices[group_idx]))
            
            # Fetch properties from the IS 800 database
            area_m2 = self.catalog.loc[cat_idx, "Area_m2"]
            r_min_m = self.catalog.loc[cat_idx, "r_min_m"]
            weight_kg_per_m = self.catalog.loc[cat_idx, "Weight_kg_m"]
            
            # Apply to all members in this specific symmetry group
            for m_id in member_ids:
                m = next((member for member in self.ts.members if member.id == m_id), None)
                if m:
                    m.A = area_m2
                    m.r_min = r_min_m
                    m.k_global_matrix = (m.E * m.A / m.L) * np.outer(m.T_vector, m.T_vector)
                    weight += m.L * weight_kg_per_m
                    
        # 2. Solve the 3D space truss
        try:
            self.ts.solve() 
        except Exception:
            # If the matrix is singular (unstable mechanism), return a massive penalty
            return 1e12 
            
        # 3. Constraints & Penalties
        penalty = 0.0
        
        # Deflection Check
        if self.ts.U_global is not None:
            max_disp = np.max(np.abs(self.ts.U_global))
            if max_disp > self.max_deflection:
                penalty += 1e9 * (max_disp / self.max_deflection)
                
        # IS 800 Stress & Buckling Check
        for m in self.ts.members:
            actual_stress = m.internal_force / m.A
            
            if actual_stress > 0: # Tension Member
                # Check against standard yield with material safety factor
                allowable_tens = self.yield_stress / 1.1 
                if actual_stress > allowable_tens:
                    penalty += 1e9 * (actual_stress / allowable_tens)
                    
            else: # Compression Member
                # Check against IS 800 buckling curve 'c'
                allowable_comp = m.get_is800_buckling_stress(self.yield_stress)
                if abs(actual_stress) > allowable_comp:
                    penalty += 1e9 * (abs(actual_stress) / allowable_comp)
                    
        return weight + penalty

    def optimize(self, pop_size=15, max_gen=100):
        """
        Runs Differential Evolution optimized for discrete integer indices.
        """
        max_index = len(self.catalog) - 1
        
        # The bounds are the top and bottom of your catalog DataFrame (0 to 19)
        bounds = [(0, max_index) for _ in range(self.num_groups)]
        
        # Force the algorithm to only guess whole numbers (discrete optimization)
        integrality = np.ones(self.num_groups)
        
        result = differential_evolution(
            self.objective_function, 
            bounds, 
            integrality=integrality,  
            strategy='best1bin', 
            popsize=pop_size, 
            maxiter=max_gen, 
            tol=0.01, 
            mutation=(0.5, 1.0), 
            recombination=0.7,
            disp=False 
        )
        
        # Retrieve final optimal catalog indices
        opt_indices = [int(round(idx)) for idx in result.x]
        
        # Re-run objective function once to get exact final weight without penalty calculations
        final_weight = self.objective_function(opt_indices) 
        
        # Create a dictionary mapping member IDs to their optimized IS Designation (e.g., M1 -> ISA 50x50x5)
        final_sections = {}
        for group_idx, member_ids in enumerate(self.member_groups):
            cat_idx = opt_indices[group_idx]
            section_name = self.catalog.loc[cat_idx, "Designation"]
            for m_id in member_ids:
                final_sections[m_id] = section_name
                
        # True if the best design found didn't trip the 1 Billion point failure penalty
        is_valid = final_weight < 1e6 
                
        return final_sections, final_weight, is_valid
