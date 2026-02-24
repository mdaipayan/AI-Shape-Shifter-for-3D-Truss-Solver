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
        self.r_min = r_min
        self.internal_force = 0.0
        
        self.dofs = self.node_i.dofs + self.node_j.dofs
        self.u_local = None
        
        # Initialize kinematics
        self.update_geometry()
        
    def update_geometry(self):
        """
        Recalculates lengths and transformation matrices. 
        Crucial for Shape Optimization when AI moves the nodes.
        """
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        self.L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if self.L < 1e-5:
            raise ValueError(f"Member {self.id} length approached zero during shape optimization.")
            
        self.l = dx / self.L
        self.m = dy / self.L
        self.n = dz / self.L
        
        self.T_vector = np.array([-self.l, -self.m, -self.n, self.l, self.m, self.n])
        self.L_current = self.L
        
        # Update the linear stiffness matrix using the new geometry
        self.k_global_matrix = (self.E * self.A / self.L) * np.outer(self.T_vector, self.T_vector)
        
    def get_k_geometric(self, current_force):
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
        if self.u_local is not None:
            self.internal_force = (self.E * self.A / self.L) * np.dot(self.T_vector, self.u_local)
        return self.internal_force

    def get_is800_buckling_stress(self, fy=250e6):
        if self.r_min <= 0:
            return fy / 1.1 
            
        KL = 1.0 * self.L 
        slenderness = KL / self.r_min
        fcc = (np.pi**2 * self.E) / (slenderness**2)
        lambda_n = np.sqrt(fy / fcc)
        alpha = 0.49 
        phi = 0.5 * (1 + alpha * (lambda_n - 0.2) + lambda_n**2)
        fcd = fy / (phi + np.sqrt(max(0, phi**2 - lambda_n**2)))
        gamma_m0 = 1.1 
        
        return min(fcd, fy) / gamma_m0

class TrussSystem:
    def __init__(self):
        self.nodes = []
        self.members = []
        self.loads = {} 
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
        
        for member in self.members:
            # Ensure geometry is updated before matrix assembly
            member.update_geometry()
            for i in range(6):
                for j in range(6):
                    self.K_global[member.dofs[i], member.dofs[j]] += member.k_global_matrix[i, j]
                    
        for dof, force in self.loads.items():
            self.F_global[dof] += force
            
        restrained_dofs = []
        for node in self.nodes:
            if node.rx: restrained_dofs.append(node.dofs[0])
            if node.ry: restrained_dofs.append(node.dofs[1])
            if node.rz: restrained_dofs.append(node.dofs[2])
            
        self.free_dofs = [i for i in range(num_dofs) if i not in restrained_dofs]
        
        self.K_reduced = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        self.F_reduced = self.F_global[self.free_dofs]
        
        if self.K_reduced.size > 0:
            cond_num = np.linalg.cond(self.K_reduced)
            if cond_num > 1e12:
                raise ValueError("Structure is unstable. Check shape optimization bounds and connectivity.")
            U_reduced = np.linalg.solve(self.K_reduced, self.F_reduced)
        else:
            U_reduced = np.array([])
            
        self.U_global = np.zeros(num_dofs)
        for idx, dof in enumerate(self.free_dofs):
            self.U_global[dof] = U_reduced[idx]
            
        R_global = np.dot(self.K_global, self.U_global) - self.F_global
        for node in self.nodes:
            node.rx_val = R_global[node.dofs[0]] if node.rx else 0.0
            node.ry_val = R_global[node.dofs[1]] if node.ry else 0.0
            node.rz_val = R_global[node.dofs[2]] if node.rz else 0.0
            
        for member in self.members:
            member.u_local = np.array([self.U_global[dof] for dof in member.dofs])
            member.calculate_force()

    def solve_nonlinear(self, load_steps=10, tolerance=1e-5, max_iter=50):
        # ... (Your existing solve_nonlinear code remains identical here) ...
        num_dofs = 3 * len(self.nodes)
        
        restrained_dofs = []
        for node in self.nodes:
            if node.rx: restrained_dofs.append(node.dofs[0])
            if node.ry: restrained_dofs.append(node.dofs[1])
            if node.rz: restrained_dofs.append(node.dofs[2])
        self.free_dofs = [i for i in range(num_dofs) if i not in restrained_dofs]
        
        F_target = np.zeros(num_dofs)
        for dof, force in self.loads.items():
            F_target[dof] += force
            
        self.U_global = np.zeros(num_dofs)
        member_forces = {m.id: 0.0 for m in self.members}
        
        for step in range(1, load_steps + 1):
            F_ext = (step / load_steps) * F_target
            
            for iteration in range(max_iter):
                K_T = np.zeros((num_dofs, num_dofs))
                F_int = np.zeros(num_dofs) 
                
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
                    
                    KE = (m.E * m.A / m.L) * np.outer(m.T_vector, m.T_vector) 
                    KG = m.get_k_geometric(member_forces[m.id])
                    K_element = KE + KG
                    
                    for i in range(6):
                        for j in range(6):
                            K_T[m.dofs[i], m.dofs[j]] += K_element[i, j]
                            
                    m.u_local = np.array([self.U_global[dof] for dof in m.dofs])
                    force = (m.E * m.A / m.L) * (m.L_current - m.L) 
                    member_forces[m.id] = force
                    
                    global_f_int = force * np.array([-m.l, -m.m, -m.n, m.l, m.m, m.n])
                    for i in range(6):
                        F_int[m.dofs[i]] += global_f_int[i]

                Residual = F_ext - F_int
                Residual_free = Residual[self.free_dofs]
                
                if np.linalg.norm(Residual_free) < tolerance:
                    break 
                    
                K_T_reduced = K_T[np.ix_(self.free_dofs, self.free_dofs)]
                delta_U_free = np.linalg.solve(K_T_reduced, Residual_free)
                
                for idx, dof in enumerate(self.free_dofs):
                    self.U_global[dof] += delta_U_free[idx]
                    
            if iteration == max_iter - 1:
                raise ValueError(f"Newton-Raphson failed to converge at load step {step}.")

        self.K_global = K_T
        self.F_global = F_target
        
        for m in self.members:
            m.internal_force = member_forces[m.id]
