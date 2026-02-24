import numpy as np

class Node:
    def __init__(self, id, x, y, z, rx=0, ry=0, rz=0):
        self.id = id
        self.user_id = id
        self.x = x
        self.y = y
        self.z = z
        
        # Support Conditions (True if restrained)
        self.rx = bool(rx)
        self.ry = bool(ry)
        self.rz = bool(rz)
        
        # Reaction Forces
        self.rx_val = 0.0
        self.ry_val = 0.0
        self.rz_val = 0.0
        
        # 3 DOFs per node in a Space Truss: [X, Y, Z]
        self.dofs = [3 * id - 3, 3 * id - 2, 3 * id - 1]

class Member:
    def __init__(self, id, node_i, node_j, E, A, r_min=0.01):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.A = A
        self.r_min = r_min  # Minimum radius of gyration (meters)
        self.internal_force = 0.0
        
        # 1. 3D Kinematics (Length)
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        self.L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if self.L == 0:
            raise ValueError(f"Member {self.id} has zero length.")
            
        # 2. Direction Cosines (l, m, n)
        self.l = dx / self.L
        self.m = dy / self.L
        self.n = dz / self.L
        
        # 3. Transformation Vector (T)
        self.T_vector = np.array([-self.l, -self.m, -self.n, self.l, self.m, self.n])
        
        # 4. Element Stiffness Matrix in Global Coordinates (6x6)
        self.k_global_matrix = (self.E * self.A / self.L) * np.outer(self.T_vector, self.T_vector)
        
        # Map element DOFs to system DOFs
        self.dofs = self.node_i.dofs + self.node_j.dofs
        self.u_local = None
        
    def get_k_geometric(self, current_force):
        """Calculates the 6x6 Geometric Stiffness Matrix (K_G)."""
        Z = np.array([
            [1 - self.l**2, -self.l*self.m, -self.l*self.n],
            [-self.l*self.m, 1 - self.m**2, -self.m*self.n],
            [-self.l*self.n, -self.m*self.n, 1 - self.n**2]
        ])
        
        KG_sub = (current_force / self.L) * Z
        
        KG = np.zeros((6, 6))
        KG[0:3, 0:3] = KG_sub
        KG[3:6, 3:6] = KG_sub
        KG[0:3, 3:6] = -KG_sub
        KG[3:6, 0:3] = -KG_sub
        
        return KG
        
    def calculate_force(self):
        """Calculates axial force. Positive = Tension, Negative = Compression."""
        if self.u_local is not None:
            self.internal_force = (self.E * self.A / self.L) * np.dot(self.T_vector, self.u_local)
        return self.internal_force

    def get_is800_buckling_stress(self, fy=250e6):
        """
        Calculates allowable compressive design stress (fcd) per IS 800:2007.
        Assumes Buckling Class 'c' (alpha = 0.49) for standard Angle Sections.
        """
        if self.r_min <= 0:
            return fy / 1.1 # Fallback safety
            
        KL = 1.0 * self.L # Effective length (K=1.0 for pinned space truss)
        slenderness = KL / self.r_min
        
        # Euler buckling stress
        fcc = (np.pi**2 * self.E) / (slenderness**2)
        
        # Non-dimensional slenderness ratio
        lambda_n = np.sqrt(fy / fcc)
        
        # Imperfection factor for Buckling Class C (Angles)
        alpha = 0.49 
        
        phi = 0.5 * (1 + alpha * (lambda_n - 0.2) + lambda_n**2)
        
        # Calculate fcd
        fcd = fy / (phi + np.sqrt(max(0, phi**2 - lambda_n**2)))
        
        # IS 800 Partial safety factor for material yielding (gamma_m0 = 1.1)
        gamma_m0 = 1.1 
        
        return min(fcd, fy) / gamma_m0

class TrussSystem:
    def __init__(self):
        self.nodes = []
        self.members = []
        self.loads = {}  # Dictionary of {dof_index: force_value}
        
        # State variables to support the pedagogical "Glass-Box" UI
        self.K_global = None
        self.F_global = None
        self.free_dofs = []
        self.K_reduced = None
        self.F_reduced = None
        self.U_global = None
        
    def solve(self):
        num_dofs = 3 * len(self.nodes)
        self.K_global = np.zeros((num_dofs, num_dofs))
        self.F_global = np.zeros(num_dofs)
        
        # 1. Assemble Global Stiffness Matrix
        for member in self.members:
            for i in range(6):
                for j in range(6):
                    self.K_global[member.dofs[i], member.dofs[j]] += member.k_global_matrix[i, j]
                    
        # 2. Assemble Load Vector
        for dof, force in self.loads.items():
            self.F_global[dof] += force
            
        # 3. Apply Boundary Conditions (Matrix Partitioning)
        restrained_dofs = []
        for node in self.nodes:
            if node.rx: restrained_dofs.append(node.dofs[0])
            if node.ry: restrained_dofs.append(node.dofs[1])
            if node.rz: restrained_dofs.append(node.dofs[2])
            
        self.free_dofs = [i for i in range(num_dofs) if i not in restrained_dofs]
        
        # Isolate the Free-Free matrix components
        self.K_reduced = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        self.F_reduced = self.F_global[self.free_dofs]
        
        # Mathematical Bulletproofing: Check for structural instability
        if self.K_reduced.size > 0:
            cond_num = np.linalg.cond(self.K_reduced)
            if cond_num > 1e12:
                raise ValueError("Structure is unstable (mechanism detected). Check boundary conditions and member connectivity.")
            
            # 4. Solve for Displacements (U_f = K_ff^-1 * F_f)
            U_reduced = np.linalg.solve(self.K_reduced, self.F_reduced)
        else:
            U_reduced = np.array([])
            
        # Reconstruct full global displacement vector
        self.U_global = np.zeros(num_dofs)
        for idx, dof in enumerate(self.free_dofs):
            self.U_global[dof] = U_reduced[idx]
            
        # 5. Calculate Support Reactions (R = K * U - F)
        R_global = np.dot(self.K_global, self.U_global) - self.F_global
        for node in self.nodes:
            node.rx_val = R_global[node.dofs[0]] if node.rx else 0.0
            node.ry_val = R_global[node.dofs[1]] if node.ry else 0.0
            node.rz_val = R_global[node.dofs[2]] if node.rz else 0.0
            
        # 6. Extract Member Forces & Local Kinematics
        for member in self.members:
            # Pull the 6 nodal displacements corresponding to this specific member
            member.u_local = np.array([self.U_global[dof] for dof in member.dofs])
            member.calculate_force()

    def solve_nonlinear(self, load_steps=10, tolerance=1e-5, max_iter=50):
        """Solves the system using the Incremental Newton-Raphson method for Geometric Non-Linearity."""
        num_dofs = 3 * len(self.nodes)
        
        # Determine free DOFs
        restrained_dofs = []
        for node in self.nodes:
            if node.rx: restrained_dofs.append(node.dofs[0])
            if node.ry: restrained_dofs.append(node.dofs[1])
            if node.rz: restrained_dofs.append(node.dofs[2])
        self.free_dofs = [i for i in range(num_dofs) if i not in restrained_dofs]
        
        # Assemble target load vector
        F_target = np.zeros(num_dofs)
        for dof, force in self.loads.items():
            F_target[dof] += force
            
        # Initialize state variables
        self.U_global = np.zeros(num_dofs)
        member_forces = {m.id: 0.0 for m in self.members}
        
        # Incremental Load Loop
        for step in range(1, load_steps + 1):
            F_ext = (step / load_steps) * F_target
            
            # Newton-Raphson Iteration Loop
            for iteration in range(max_iter):
                # 1. Build Tangent Stiffness Matrix (K_T = K_E + K_G)
                K_T = np.zeros((num_dofs, num_dofs))
                F_int = np.zeros(num_dofs) # Internal force vector
                
                for m in self.members:
                    # Update kinematics based on CURRENT displaced geometry
                    n_i = self.nodes[m.node_i.id - 1]
                    n_j = self.nodes[m.node_j.id - 1]
                    
                    dx = (n_j.x + self.U_global[n_j.dofs[0]]) - (n_i.x + self.U_global[n_i.dofs[0]])
                    dy = (n_j.y + self.U_global[n_j.dofs[1]]) - (n_i.y + self.U_global[n_i.dofs[1]])
                    dz = (n_j.z + self.U_global[n_j.dofs[2]]) - (n_i.z + self.U_global[n_i.dofs[2]])
                    
                    m.L_current = np.sqrt(dx**2 + dy**2 + dz**2)
                    m.l, m.m, m.n = dx/m.L_current, dy/m.L_current, dz/m.L_current
                    m.T_vector = np.array([-m.l, -m.m, -m.n, m.l, m.m, m.n])
                    
                    # Recalculate K_E and K_G
                    KE = (m.E * m.A / m.L) * np.outer(m.T_vector, m.T_vector) 
                    KG = m.get_k_geometric(member_forces[m.id])
                    K_element = KE + KG
                    
                    # Assemble global K_T
                    for i in range(6):
                        for j in range(6):
                            K_T[m.dofs[i], m.dofs[j]] += K_element[i, j]
                            
                    # Calculate internal forces using exact physical stretch (FIXED)
                    m.u_local = np.array([self.U_global[dof] for dof in m.dofs])
                    force = (m.E * m.A / m.L) * (m.L_current - m.L) 
                    member_forces[m.id] = force
                    
                    # Map element internal forces to global internal force vector
                    global_f_int = force * np.array([-m.l, -m.m, -m.n, m.l, m.m, m.n])
                    for i in range(6):
                        F_int[m.dofs[i]] += global_f_int[i]

                # 2. Calculate Unbalanced Forces (Residuals)
                Residual = F_ext - F_int
                Residual_free = Residual[self.free_dofs]
                
                # Check Convergence
                if np.linalg.norm(Residual_free) < tolerance:
                    break 
                    
                # 3. Solve for Displacement Increment
                K_T_reduced = K_T[np.ix_(self.free_dofs, self.free_dofs)]
                delta_U_free = np.linalg.solve(K_T_reduced, Residual_free)
                
                # 4. Update Displacements
                for idx, dof in enumerate(self.free_dofs):
                    self.U_global[dof] += delta_U_free[idx]
                    
            if iteration == max_iter - 1:
                raise ValueError(f"Newton-Raphson failed to converge at load step {step}.")

        # Finalize final states
        self.K_global = K_T
        self.F_global = F_target
        
        for m in self.members:
            m.internal_force = member_forces[m.id]
