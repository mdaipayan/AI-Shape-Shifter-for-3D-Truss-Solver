import numpy as np
from scipy.optimize import differential_evolution
import copy
from is_catalog import get_isa_catalog

class TrussOptimizer:
    def __init__(self, base_truss, member_groups=None, yield_stress=250e6, max_deflection=0.05):
        """
        Discrete AI Optimizer using IS 800 Catalog and Member Grouping.
        """
        self.ts = copy.deepcopy(base_truss) 
        self.catalog = get_isa_catalog()
        self.yield_stress = yield_stress
        self.max_deflection = max_deflection
        self.history = [] # Array to track convergence history
        
        # Setup Grouping (Symmetry)
        if member_groups is None:
            self.member_groups = [[m.id] for m in self.ts.members]
        else:
            self.member_groups = member_groups
            
        self.num_groups = len(self.member_groups)

    def objective_function(self, group_indices):
        """Evaluates structural weight and IS 800 penalties."""
        weight = 0.0
        
        # 1. Apply catalog properties
        for group_idx, member_ids in enumerate(self.member_groups):
            cat_idx = int(round(group_indices[group_idx]))
            area_m2 = self.catalog.loc[cat_idx, "Area_m2"]
            r_min_m = self.catalog.loc[cat_idx, "r_min_m"]
            weight_kg_per_m = self.catalog.loc[cat_idx, "Weight_kg_m"]
            
            for m_id in member_ids:
                # FIX: Scoping issue resolved by using 'member.id' and 'mbr'
                mbr = next((member for member in self.ts.members if member.id == m_id), None)
                if mbr:
                    mbr.A = area_m2
                    mbr.r_min = r_min_m
                    mbr.k_global_matrix = (mbr.E * mbr.A / mbr.L) * np.outer(mbr.T_vector, mbr.T_vector)
                    weight += mbr.L * weight_kg_per_m
                    
        # 2. Solve the structure
        try:
            self.ts.solve() 
        except Exception:
            return 1e12 # Mechanism penalty
            
        # 3. Constraints & Penalties
        penalty = 0.0
        
        if self.ts.U_global is not None:
            max_disp = np.max(np.abs(self.ts.U_global))
            if max_disp > self.max_deflection:
                penalty += 1e9 * (max_disp / self.max_deflection)**2
                
        for m in self.ts.members:
            actual_stress = m.internal_force / m.A
            if actual_stress > 0: # Tension
                allowable_tens = self.yield_stress / 1.1 
                if actual_stress > allowable_tens:
                    # Exponential scaling for smoother gradient mapping
                    penalty += 1e9 * (actual_stress / allowable_tens)**2
            else: # Compression (IS 800 Buckling)
                allowable_comp = m.get_is800_buckling_stress(self.yield_stress)
                if abs(actual_stress) > allowable_comp:
                    # Exponential scaling for smoother gradient mapping
                    penalty += 1e9 * (abs(actual_stress) / allowable_comp)**2
                    
        return weight + penalty

    def _callback(self, xk, convergence=None):
        """
        SciPy callback function. Fires at the end of every generation.
        xk contains the best IS Catalog indices found so far.
        """
        best_val = self.objective_function(xk)
        self.history.append(best_val)

    def optimize(self, pop_size=15, max_gen=100):
        """Runs Differential Evolution and returns history tracking."""
        self.history = [] # Reset history on new run
        max_index = len(self.catalog) - 1
        bounds = [(0, max_index) for _ in range(self.num_groups)]
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
            callback=self._callback, # Attach the tracker
            disp=False 
        )
        
        opt_indices = [int(round(idx)) for idx in result.x]
        final_weight = self.objective_function(opt_indices) 
        
        final_sections = {}
        for group_idx, member_ids in enumerate(self.member_groups):
            cat_idx = opt_indices[group_idx]
            section_name = self.catalog.loc[cat_idx, "Designation"]
            for m_id in member_ids:
                final_sections[m_id] = section_name
                
        is_valid = final_weight < 1e6 
                
        # Returning 4 variables, including the history array
        return final_sections, final_weight, is_valid, self.history
