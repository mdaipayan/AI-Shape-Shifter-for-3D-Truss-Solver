import numpy as np
from scipy.optimize import differential_evolution
import copy
from is_catalog import get_isa_catalog

class TrussOptimizer:
    def __init__(self, base_combos, is_nonlinear=False, load_steps=10, member_groups=None, shape_bounds=None, yield_stress=250e6, max_deflection=0.05):
        """
        Combined Shape and Sizing Optimizer.
        shape_bounds: dict mapping Node_ID to [dx_min, dx_max, dy_min, dy_max, dz_min, dz_max]
        """
        self.combos = [copy.deepcopy(ts) for ts in base_combos]
        self.is_nonlinear = is_nonlinear
        self.load_steps = load_steps
        
        self.catalog = get_isa_catalog()
        self.yield_stress = yield_stress
        self.max_deflection = max_deflection
        self.history = [] 
        
        ref_ts = self.combos[0]
        if member_groups is None:
            self.member_groups = [[m.id] for m in ref_ts.members]
        else:
            self.member_groups = member_groups
            
        self.num_groups = len(self.member_groups)
        
        # New: Shape Optimization setup
        self.shape_bounds = shape_bounds if shape_bounds else {}
        self.num_shape_vars = len(self.shape_bounds) * 3
        
        # Store original coordinates to apply relative shifts cleanly
        self.base_coords = {}
        for n in ref_ts.nodes:
            self.base_coords[n.id] = (n.x, n.y, n.z)

    def objective_function(self, xk):
        """Evaluates MINLP fitness (Sizing + Shape)."""
        weight = 0.0
        
        # Unpack the AI chromosome
        group_indices = xk[:self.num_groups]
        shape_vars = xk[self.num_groups:]
        
        # 1. Fetch catalog properties for sizing
        group_props = {}
        for group_idx, member_ids in enumerate(self.member_groups):
            cat_idx = int(round(group_indices[group_idx]))
            area_m2 = self.catalog.loc[cat_idx, "Area_m2"]
            r_min_m = self.catalog.loc[cat_idx, "r_min_m"]
            weight_kg_per_m = self.catalog.loc[cat_idx, "Weight_kg_m"]
            group_props[group_idx] = {'A': area_m2, 'r': r_min_m, 'w': weight_kg_per_m}

        max_nodal_disp = 0.0
        member_stresses = {m.id: {'tension': 0.0, 'compression': 0.0} for m in self.combos[0].members}
        
        # 2. Solve ALL combinations for this specific Shape + Sizing guess
        for ts in self.combos:
            # A) Apply Shape Shifts to Nodes
            shape_idx = 0
            for n_id in self.shape_bounds.keys():
                dx = shape_vars[shape_idx]
                dy = shape_vars[shape_idx+1]
                dz = shape_vars[shape_idx+2]
                shape_idx += 3
                
                node = next((n for n in ts.nodes if n.id == n_id), None)
                if node:
                    base_x, base_y, base_z = self.base_coords[n_id]
                    node.x = base_x + dx
                    node.y = base_y + dy
                    node.z = base_z + dz
            
            # B) Apply Sizing and Update Geometry
            for group_idx, member_ids in enumerate(self.member_groups):
                props = group_props[group_idx]
                for m_id in member_ids:
                    mbr = next((m for m in ts.members if m.id == m_id), None)
                    if mbr:
                        mbr.A = props['A']
                        mbr.r_min = props['r']
                        # CRITICAL: Recompute L, l, m, n based on shifted nodes
                        try:
                            mbr.update_geometry()
                        except ValueError:
                            return 1e12 # AI caused members to overlap/invert
            
            # C) Solve the Matrices
            try:
                if self.is_nonlinear:
                    ts.solve_nonlinear(load_steps=self.load_steps)
                else:
                    ts.solve()
            except Exception:
                return 1e12 # Mechanism penalty
                
            # D) Extract Envelopes
            if ts.U_global is not None:
                current_max_disp = np.max(np.abs(ts.U_global))
                if current_max_disp > max_nodal_disp:
                    max_nodal_disp = current_max_disp
                    
            for mbr in ts.members:
                actual_stress = mbr.internal_force / mbr.A
                if actual_stress > 0: 
                    member_stresses[mbr.id]['tension'] = max(member_stresses[mbr.id]['tension'], actual_stress)
                else: 
                    member_stresses[mbr.id]['compression'] = min(member_stresses[mbr.id]['compression'], actual_stress)

        # 3. Calculate Weight based on the newly morphed lengths
        ref_ts = self.combos[0]
        for group_idx, member_ids in enumerate(self.member_groups):
            w_per_m = group_props[group_idx]['w']
            for m_id in member_ids:
                mbr = next((m for m in ref_ts.members if m.id == m_id), None)
                if mbr:
                    weight += mbr.L * w_per_m

        # 4. Constraints & Penalties
        penalty = 0.0
        if max_nodal_disp > self.max_deflection:
            penalty += 1e9 * (max_nodal_disp / self.max_deflection)**2
            
        allowable_tens = self.yield_stress / 1.1 
        
        for mbr in ref_ts.members:
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
        
        # 1. Define Sizing Bounds
        bounds = [(0, max_index) for _ in range(self.num_groups)]
        integrality = [1] * self.num_groups # 1 means integer (discrete)
        
        # 2. Append Shape Bounds
        for n_id, b in self.shape_bounds.items():
            bounds.extend([(b[0], b[1]), (b[2], b[3]), (b[4], b[5])])
            integrality.extend([0, 0, 0]) # 0 means float (continuous)
        
        result = differential_evolution(
            self.objective_function, 
            bounds, 
            integrality=np.array(integrality),  
            strategy='best1bin', 
            popsize=pop_size, 
            maxiter=max_gen, 
            tol=0.01, 
            mutation=(0.5, 1.0), 
            recombination=0.7,
            callback=self._callback, 
            disp=False 
        )
        
        # Extract Results
        opt_indices = [int(round(idx)) for idx in result.x[:self.num_groups]]
        opt_shape = result.x[self.num_groups:]
        final_weight = self.objective_function(result.x) 
        
        final_sections = {}
        for group_idx, member_ids in enumerate(self.member_groups):
            cat_idx = opt_indices[group_idx]
            section_name = self.catalog.loc[cat_idx, "Designation"]
            for m_id in member_ids:
                final_sections[m_id] = section_name
                
        final_node_shifts = {}
        shape_idx = 0
        for n_id in self.shape_bounds.keys():
            final_node_shifts[n_id] = {
                'dx': opt_shape[shape_idx],
                'dy': opt_shape[shape_idx+1],
                'dz': opt_shape[shape_idx+2]
            }
            shape_idx += 3
                
        is_valid = final_weight < 1e6 
        return final_sections, final_node_shifts, final_weight, is_valid, self.history
