import numpy as np
from scipy.optimize import differential_evolution
import copy
from is_catalog import get_isa_catalog

class TrussOptimizer:
    def __init__(self, base_combos, is_nonlinear=False, load_steps=10, member_groups=None, yield_stress=250e6, max_deflection=0.05):
        """
        Discrete AI Optimizer using IS 800 Catalog and Member Grouping.
        Evaluates the worst-case Absolute Max/Min Envelope across multiple load combinations.
        """
        # Deepcopy all combinations so we don't accidentally overwrite the UI's solved state
        self.combos = [copy.deepcopy(ts) for ts in base_combos]
        self.is_nonlinear = is_nonlinear
        self.load_steps = load_steps
        
        self.catalog = get_isa_catalog()
        self.yield_stress = yield_stress
        self.max_deflection = max_deflection
        self.history = [] 
        
        # Setup Grouping (Symmetry) based on the first combination's topology
        ref_ts = self.combos[0]
        if member_groups is None:
            self.member_groups = [[m.id] for m in ref_ts.members]
        else:
            self.member_groups = member_groups
            
        self.num_groups = len(self.member_groups)

    def objective_function(self, group_indices):
        """Evaluates structural weight and IS 800 penalties across the load envelope."""
        weight = 0.0
        
        # 1. Fetch catalog properties for this specific AI candidate design
        group_props = {}
        for group_idx, member_ids in enumerate(self.member_groups):
            cat_idx = int(round(group_indices[group_idx]))
            area_m2 = self.catalog.loc[cat_idx, "Area_m2"]
            r_min_m = self.catalog.loc[cat_idx, "r_min_m"]
            weight_kg_per_m = self.catalog.loc[cat_idx, "Weight_kg_m"]
            group_props[group_idx] = {'A': area_m2, 'r': r_min_m, 'w': weight_kg_per_m}
            
            # Calculate steel weight using just the reference geometry
            for m_id in member_ids:
                mbr = next((m for m in self.combos[0].members if m.id == m_id), None)
                if mbr:
                    weight += mbr.L * weight_kg_per_m

        max_nodal_disp = 0.0
        # Initialize an envelope tracker for every member ID
        member_stresses = {m.id: {'tension': 0.0, 'compression': 0.0} for m in self.combos[0].members}
        
        # 2. Solve ALL combinations to build the Envelope
        for ts in self.combos:
            # Apply AI-selected cross-sections to this specific combination
            for group_idx, member_ids in enumerate(self.member_groups):
                props = group_props[group_idx]
                for m_id in member_ids:
                    mbr = next((m for m in ts.members if m.id == m_id), None)
                    if mbr:
                        mbr.A = props['A']
                        mbr.r_min = props['r']
                        mbr.k_global_matrix = (mbr.E * mbr.A / mbr.L) * np.outer(mbr.T_vector, mbr.T_vector)
            
            # Solve the structural matrices
            try:
                if self.is_nonlinear:
                    ts.solve_nonlinear(load_steps=self.load_steps)
                else:
                    ts.solve()
            except Exception:
                return 1e12 # Mechanism penalty
                
            # Extract Max Displacements
            if ts.U_global is not None:
                current_max_disp = np.max(np.abs(ts.U_global))
                if current_max_disp > max_nodal_disp:
                    max_nodal_disp = current_max_disp
                    
            # Extract Internal Forces into the Envelope
            for mbr in ts.members:
                actual_stress = mbr.internal_force / mbr.A
                if actual_stress > 0: # Tension
                    member_stresses[mbr.id]['tension'] = max(member_stresses[mbr.id]['tension'], actual_stress)
                else: # Compression (Negative value)
                    member_stresses[mbr.id]['compression'] = min(member_stresses[mbr.id]['compression'], actual_stress)

        # 3. Constraints & Penalties evaluated against the Peak Envelope
        penalty = 0.0
        
        if max_nodal_disp > self.max_deflection:
            penalty += 1e9 * (max_nodal_disp / self.max_deflection)**2
            
        allowable_tens = self.yield_stress / 1.1 
        
        # We can use the reference geometry to calculate buckling limits
        for mbr in self.combos[0].members:
            peak_tension = member_stresses[mbr.id]['tension']
            peak_compression = abs(member_stresses[mbr.id]['compression'])
            
            if peak_tension > allowable_tens:
                penalty += 1e9 * (peak_tension / allowable_tens)**2
                
            allowable_comp = mbr.get_is800_buckling_stress(self.yield_stress)
            if peak_compression > allowable_comp:
                penalty += 1e9 * (peak_compression / allowable_comp)**2
                
        return weight + penalty

    def _callback(self, xk, convergence=None):
        best_val = self.objective_function(xk)
        self.history.append(best_val)

    def optimize(self, pop_size=15, max_gen=100):
        self.history = [] 
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
            callback=self._callback, 
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
        return final_sections, final_weight, is_valid, self.history
