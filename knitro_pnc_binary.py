import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from knitro import *
from scipy import sparse
from scipy.sparse import linalg
from src.Truss_lex import TrussStructure

# Nonlinear Optimization using knitro (Pnc)

class TrussOptimizerBinary:
    def __init__(self, truss, force_vector, volume_fraction=0.5):
        self.truss = truss
        self.force_vector = force_vector
        self.n_bars = len(self.truss.bars)
        self.n_dofs = self.truss.total_dofs

        # Identify design edges and corresponding indices
        self.design_edge_indices = []
        for i, edge in enumerate(self.truss.edges):
            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_norm in self.truss.design_edges_set:
                self.design_edge_indices.append(i)

        self.n_design_vars = len(self.design_edge_indices)
        print(f"Number of design variables: {self.n_design_vars} (out of {self.n_bars} total bars)")

        # Volume constraint parameters
        self.design_volumes = self._compute_design_volumes()
        self.total_design_volume = sum(self.design_volumes)
        self.target_volume = volume_fraction * self.total_design_volume

        # Pre-compute stiffness matrices
        self.K_virtual_array = []
        self.K_actual_array = []
        self.K_diff_array = []
        self._precompute_stiffness_matrices()

        # Map force vector to reduced space
        self.force = np.zeros(self.n_dofs)
        for i in range(len(self.force_vector)):
            if i in self.truss.map_dof_entire2subset:
                mapped_idx = self.truss.map_dof_entire2subset[i]
                self.force[mapped_idx] = self.force_vector[i]

    def _precompute_stiffness_matrices(self):
        """Precompute stiffness matrices for each bar."""
        print("Precomputing stiffness matrices...")
        for i in range(self.n_bars):
            k_actual, k_virtual = self.truss.get_bar_stiffness_matrix(i)
            self.K_virtual_array.append(k_virtual)
            self.K_actual_array.append(k_actual)
            self.K_diff_array.append(k_actual - k_virtual)

    def _compute_design_volumes(self):
        """Compute volume of each design bar."""
        volumes = []
        for i in self.design_edge_indices:
            bar = self.truss.bars[i]
            volume = bar.section_.Ax_ * bar.length_
            volumes.append(volume)
        return np.array(volumes)

    def _assemble_global_stiffness(self, design_rho):
        """Assemble global stiffness matrix for given design densities."""
        # Start with virtual stiffness for all elements (using sparse matrix addition)
        K_global = self.K_virtual_array[0].copy()
        for i in range(1, len(self.K_virtual_array)):
            K_global += self.K_virtual_array[i]

        # Convert to CSR format for efficient operations
        K_global = K_global.tocsr()
        # Add fixed stiffness difference for non-design edges
        for i in range(self.n_bars):
            if i not in self.design_edge_indices:
                K_global += self.K_diff_array[i]  # Full stiffness for non-design edges (rho=1)

        # Add weighted stiffness difference for design edges
        for idx, i in enumerate(self.design_edge_indices):
            K_global += (design_rho[idx]+1e-6) * self.K_diff_array[i] # prevent singularity
        return K_global

    # minimize f^T*K(rho)^-1*f where K(rho) = sum(rho_i * K_i)
    def callbackEvalFC(self, kc, cb, evalRequest, evalResult, userParams):
        """Evaluate objective function only."""
        if evalRequest.type != KN_RC_EVALFC:
            print(f"*** callbackEvalFC incorrectly called with eval type {evalRequest.type}")
            return -1

        # These are ONLY the design variables (not all bars)
        design_rho = evalRequest.x

        # Calculate stiffness matrix and solve for displacements
        K = self._assemble_global_stiffness(design_rho)
        try:
            u = linalg.spsolve(K, self.force) # TODO: To avoid singularity
            # Objective: compliance (f^T u)
            compliance = self.force.dot(u)
            evalResult.obj = compliance
            return 0
        except Exception as e:
            print(f"Error in objective evaluation: {e}")
            # Return a large compliance value if matrix is singular
            evalResult.obj = 1e10
            return 0

    # Nonlinear terms Jacobian part
    # ∂(f^TK(rho)^-1f)/∂(rho_i) = f^T* ∂(K(rho)^-1)/∂(rho_i) * f
    # ∂(K(rho)^-1)/∂(rho_i) = -K^-1 * K_i * K^-1
    # f^T* ∂(K(rho)^-1)/∂(rho_i) * f = -(f^T * K^-1) * K_i * (K^-1 * f) = -u^T * K_i * u
    def callbackEvalGA(self, kc, cb, evalRequest, evalResult, userParams):
        """Evaluate gradients of objective only."""
        if evalRequest.type != KN_RC_EVALGA:
            print(f"*** callbackEvalGA incorrectly called with eval type {evalRequest.type}")
            return -1

        design_rho = evalRequest.x

        # Calculate stiffness matrix and solve for displacements
        K = self._assemble_global_stiffness(design_rho)
        try:
            u = linalg.spsolve(K, self.force)

            # Gradient of compliance with respect to each design variable
            for idx, i in enumerate(self.design_edge_indices):
                # Sensitivity is -u^T (dK/drho_i) u
                dK_drho = self.K_diff_array[i]
                sensitivity = -u.dot(dK_drho.dot(u))
                evalResult.objGrad[idx] = sensitivity

            return 0
        except Exception as e:
            print(f"Error in gradient evaluation: {e}")
            # Return zeros if matrix is singular
            for idx in range(len(self.design_edge_indices)):
                evalResult.objGrad[idx] = 0.0
            return 0

    # nonlinear term Hessian part
    def callbackEvalH(self, kc, cb, evalRequest, evalResult, userParams):
        """Evaluate Hessian of the Lagrangian."""
        if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
            print(f"*** callbackEvalH incorrectly called with eval type {evalRequest.type}")
            return -1

        design_rho = evalRequest.x
        sigma = evalRequest.sigma
        lambda_ = evalRequest.lambda_

        # We only need to compute the Hessian of the objective (compliance)
        # since the constraint Hessian is zero (volume constraint is linear)
        K = self._assemble_global_stiffness(design_rho)

        try:
            u = linalg.spsolve(K, self.force)

            # Fill Hessian entries
            # Note: Knitro expects the upper triangular part only (including diagonal)
            hess_idx = 0
            n_design = len(self.design_edge_indices)

            for i_idx in range(n_design):
                i = self.design_edge_indices[i_idx]
                K_i = self.K_diff_array[i]
                K_i_u = K_i.dot(u)

                # Diagonal Hessian elements (i,i)
                # This handles the case when j=i in the formula
                # u^T * (K_i * K^-1 * K_i + K_i * K^-1 * K_i) * u = 2 * u^T * K_i * K^-1 * K_i * u
                temp = linalg.spsolve(K, K_i_u)
                hess_ii = sigma * 2 * u.dot(K_i.dot(temp))
                evalResult.hess[hess_idx] = hess_ii
                hess_idx += 1

                # Off-diagonal Hessian elements (i,j) for j>i
                for j_idx in range(i_idx + 1, n_design):
                    j = self.design_edge_indices[j_idx]
                    K_j = self.K_diff_array[j]

                    # Compute u^T * (K_j * K^-1 * K_i + K_i * K^-1 * K_j) * u
                    K_j_u = K_j.dot(u)
                    temp1 = linalg.spsolve(K, K_i_u)  # K^-1 * K_i * u
                    temp2 = linalg.spsolve(K, K_j_u)  # K^-1 * K_j * u

                    hess_ij = sigma * (u.dot(K_j.dot(temp1)) + u.dot(K_i.dot(temp2)))
                    evalResult.hess[hess_idx] = hess_ij
                    hess_idx += 1

            return 0
        except Exception as e:
            print(f"Error in Hessian evaluation: {e}")
            # Return zeros for all Hessian entries
            for idx in range(len(evalResult.hess)):
                evalResult.hess[idx] = 0.0
            return 0

    def process_node_callback(self, kcSub, x, lambda_, userParams):
        """Monitor MIP node progress."""
        try:
            # Get MIP solution information
            numNodes = KN_get_mip_number_nodes(kcSub)

            if numNodes % 10 == 0:  # Only print every 10 nodes to avoid excessive output
                relaxBound = KN_get_mip_relaxation_bnd(kcSub)
                nodeObj = KN_get_obj_value(kcSub)
                print(f"Node {numNodes}: obj = {nodeObj:.4e}, bound = {relaxBound:.4e}")

                try:
                    incObj = KN_get_mip_incumbent_obj(kcSub)
                    absGap = KN_get_mip_abs_gap(kcSub)
                    relGap = KN_get_mip_rel_gap(kcSub)
                    print(f"  Incumbent = {incObj:.4e}, abs gap = {absGap:.4e}, rel gap = {relGap:.4e}")
                except:
                    print("  No incumbent solution found yet")
        except Exception as e:
            print(f"Error in node callback: {e}")

        return 0

    def solve_binary_optimization(self, warm_start=None, time_limit=3600):
        """Solve the binary compliance minimization problem using Knitro."""
        print(f"Starting binary optimization with Knitro...")
        print(f"Target volume fraction: {self.target_volume / self.total_design_volume:.2f}")

        try:
            kc = KN_new()
        except:
            print("Failed to find a valid Knitro license.")
            return None

            # Set solver parameters - KEEPING CONSISTENT WITH RELAXED VERSION
        KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_ITER)
        KN_set_int_param(kc, KN_PARAM_OUTMODE, KN_OUTMODE_SCREEN)
        # Use the same algorithm as in relaxed version
        KN_set_int_param(kc, KN_PARAM_ALGORITHM, KN_ALG_BAR_DIRECT)  # Interior-point with direct linear solver
        # Same Hessian handling
        KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)
        # Same tolerances as relaxed version
        KN_set_double_param(kc, KN_PARAM_FEASTOL, 1e-6)  # Use same feasibility tolerance
        KN_set_double_param(kc, KN_PARAM_OPTTOL, 1e-6)  # Use same optimality tolerance
        # MIP specific parameters - can keep these as they don't affect relaxed version
        KN_set_int_param(kc, KN_PARAM_MIP_METHOD, KN_MIP_METHOD_BB)  # Branch and bound for MIP
        KN_set_int_param(kc, KN_PARAM_MIP_OUTLEVEL, KN_MIP_OUTLEVEL_ITERS)
        KN_set_double_param(kc, KN_PARAM_MIP_INTGAPREL, 0.05)  # 5% relative gap tolerance
        KN_set_double_param(kc, KN_PARAM_MAXTIMECPU, time_limit)  # Time limit in seconds

        # Add variables (design densities) - ONLY for design edges
        KN_add_vars(kc, self.n_design_vars)

        # Set variables as binary variables
        KN_set_var_types(kc, xTypes=[KN_VARTYPE_BINARY] * self.n_design_vars)

        # Set variable bounds (binary variables bounds are already [0,1])
        # Setting bounds is redundant for binaries but included for clarity
        KN_set_var_lobnds(kc, xLoBnds=[0.0] * self.n_design_vars)
        KN_set_var_upbnds(kc, xUpBnds=[1.0] * self.n_design_vars)

        # Add volume constraint as a LINEAR constraint
        KN_add_cons(kc, 1)
        # Set a one-sided inequality constraint: volume ≤ target_volume
        KN_set_con_upbnds(kc, cUpBnds=[self.target_volume])
        KN_set_con_lobnds(kc, cLoBnds=[0.0])

        # Set the linear coefficients for the volume constraint
        for i in range(self.n_design_vars):
            KN_add_con_linear_struct(kc,
                                     indexCons=[0],
                                     indexVars=[i],
                                     coefs=[self.design_volumes[i]])

        # Set optimization goal (minimize compliance)
        KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)

        # Register callbacks for nonlinear objective only
        cb = KN_add_eval_callback(kc, evalObj=True, indexCons=[],
                                  funcCallback=self.callbackEvalFC)

        # Set gradient structure for objective
        objGradIndexVars = list(range(self.n_design_vars))

        KN_set_cb_grad(kc, cb,
                       objGradIndexVars=objGradIndexVars,
                       jacIndexCons=[],
                       jacIndexVars=[],
                       gradCallback=self.callbackEvalGA)

        # Set Hessian structure (upper triangle, including diagonal)
        hessIndexVars1 = []
        hessIndexVars2 = []

        for i in range(self.n_design_vars):
            # Diagonal element
            hessIndexVars1.append(i)
            hessIndexVars2.append(i)

            # Off-diagonal elements
            for j in range(i + 1, self.n_design_vars):
                hessIndexVars1.append(i)
                hessIndexVars2.append(j)

        KN_set_cb_hess(kc, cb,
                       hessIndexVars1=hessIndexVars1,
                       hessIndexVars2=hessIndexVars2,
                       hessCallback=self.callbackEvalH)

        # Set MIP node callback
        KN_set_mip_node_callback(kc, self.process_node_callback)

        # Initial point - either from warm start or default to midpoint
        if warm_start is not None:
            # Round continuous solution to get a feasible starting point for binary problem
            binary_init = np.round(warm_start).clip(0, 1)
            KN_set_var_primal_init_values(kc, xInitVals=binary_init)
        else:
            # Default starting point
            initial_values = np.ones(self.n_design_vars) * self.target_volume / self.total_design_volume
            KN_set_var_primal_init_values(kc, xInitVals=initial_values)

        # Solve the problem
        try:
            nStatus = KN_solve(kc)

            # Get solution
            status, obj_val, design_rho, lambda_ = KN_get_solution(kc)

            print(f"Optimization completed with status {nStatus}")
            print(f"Optimal compliance: {obj_val}")
            print(f"Volume constraint value: {np.sum(self.design_volumes * design_rho)}/{self.target_volume}")

            # Expand solution to include all bars (setting non-design bars to 1.0)
            full_rho = np.ones(self.n_bars)  # Default density 1.0 for non-design bars
            for idx, i in enumerate(self.design_edge_indices):
                full_rho[i] = design_rho[idx]

            # Free Knitro instance
            KN_free(kc)

            return full_rho

        except Exception as e:
            print(f"Optimization error: {e}")
            KN_free(kc)
            return None


class TrussVisualizer:
    def __init__(self, truss_original, truss_optimized):
        self.truss_original = truss_original
        self.truss_optimized = truss_optimized
        self.show_deformed = False
        self.consider_selfweight = False
        self.force_x = 0.0
        self.force_y = -0.005  # Default vertical force
        self.force_z = 0.0
        self.optimized_densities = None
        self.volume_fraction = 0.5

        # Initialize force application point
        self.force_node_id = self.truss_original.nx * self.truss_original.ny * (
                    self.truss_original.nz - 1) + self.truss_original.nx - 1
        self.force_dof_x = 6 * self.force_node_id + 0
        self.force_dof_y = 6 * self.force_node_id + 1
        self.force_dof_z = 6 * self.force_node_id + 2

    def combine_meshes(self, V_list, F_list, disp=None):
        total_vertices = np.sum([v.shape[0] for v in V_list])
        total_faces = np.sum([f.shape[0] for f in F_list])
        V = np.zeros((total_vertices, 3))
        F = np.zeros((total_faces, 3), dtype=int)
        C = np.zeros((total_faces, 3))

        v_offset = 0
        f_offset = 0
        for i in range(len(V_list)):
            v = V_list[i]
            f = F_list[i]
            V[v_offset:v_offset + v.shape[0]] = v
            F[f_offset:f_offset + f.shape[0]] = f + v_offset

            if disp is not None:
                C[f_offset:f_offset + f.shape[0]] = disp[i]
            else:
                if self.optimized_densities is not None:
                    density = self.optimized_densities[i]
                    C[f_offset:f_offset + f.shape[0]] = np.array([0.2 + 0.6 * density, 0.2 + 0.6 * density, 0.8])
                else:
                    C[f_offset:f_offset + f.shape[0]] = np.array([0.8, 0.8, 0.8])

            v_offset += v.shape[0]
            f_offset += f.shape[0]

        return V, F, C

    def update_visualization_original(self):
        # Get complete force vector including self-weight if enabled
        eforce = self.get_force_vector()

        self.truss_original.consider_selfweight = self.consider_selfweight
        displacement, success, message = self.truss_original.solve_elasticity(eforce)

        if success:
            max_disp = np.max(np.abs(displacement))
            disp_color = self.truss_original.get_deformed_bar_displacement_colors(displacement, max_disp)
            VD_list, FD_list = self.truss_original.get_deformed_bar_geometry(displacement)
            Vd, Fd, Cd = self.combine_meshes(VD_list, FD_list, disp_color)

            self.ps_mesh_deformed_original.update_vertex_positions(Vd)
            self.ps_mesh_deformed_original.add_color_quantity("Displacement", Cd, defined_on='faces', enabled=True)
            self.ps_mesh_deformed_original.set_enabled(self.show_deformed)
            self.ps_mesh_static_original.set_enabled(not self.show_deformed)

            force_point = self.truss_original.nodes[self.force_node_id] + displacement[
                                                                          6 * self.force_node_id:6 * self.force_node_id + 3]
            self.ps_force_pt.update_point_positions(force_point.reshape(1, 3))

            force_vec = np.array([self.force_x, self.force_y, self.force_z])
            self.ps_force_pt.add_vector_quantity("single point force",
                                                 force_vec.reshape(1, 3),
                                                 radius=0.005,
                                                 enabled=self.show_deformed,
                                                 color=(1.0, 0, 0))
        else:
            print("Elasticity solution failed:", message)

    def update_static_visualization_optimized(self):
        # Update static optimized mesh without computing displacement (optimization already done)
        Vs_list, Fs_list = self.truss_optimized.get_bar_geometry()
        Vs, Fs, Cs = self.combine_meshes(Vs_list, Fs_list)
        self.ps_mesh_static_optimized.update_vertex_positions(Vs)
        self.ps_mesh_static_optimized.add_color_quantity("Color", Cs, defined_on='faces')

    def update_visualization_optimized(self):
        # Get complete force vector including self-weight if enabled
        eforce = self.get_force_vector()

        self.truss_optimized.consider_selfweight = self.consider_selfweight
        displacement, success, message = self.truss_optimized.solve_elasticity(eforce)

        if success:
            max_disp = np.max(np.abs(displacement))
            disp_color = self.truss_optimized.get_deformed_bar_displacement_colors(displacement, max_disp)
            VD_list, FD_list = self.truss_optimized.get_deformed_bar_geometry(displacement)
            Vd, Fd, Cd = self.combine_meshes(VD_list, FD_list, disp_color)

            self.ps_mesh_deformed_optimized.update_vertex_positions(Vd)
            self.ps_mesh_deformed_optimized.add_color_quantity("Displacement", Cd, defined_on='faces', enabled=True)
            self.ps_mesh_deformed_optimized.set_enabled(self.show_deformed)
            self.ps_mesh_static_optimized.set_enabled(not self.show_deformed)

            force_point = self.truss_optimized.nodes[self.force_node_id] + displacement[
                                                                           6 * self.force_node_id:6 * self.force_node_id + 3]
            self.ps_force_pt.update_point_positions(force_point.reshape(1, 3))

            force_vec = np.array([self.force_x, self.force_y, self.force_z])
            self.ps_force_pt.add_vector_quantity("single point force",
                                                 force_vec.reshape(1, 3),
                                                 radius=0.005,
                                                 enabled=self.show_deformed,
                                                 color=(1.0, 0, 0))
        else:
            print("Elasticity solution failed:", message)

    def setup_visualization(self):
        ps.init()
        ps.set_ground_plane_mode("shadow_only")

        # Static mesh
        Vs_list, Fs_list = self.truss_original.get_bar_geometry()
        Vs, Fs, Cs = self.combine_meshes(Vs_list, Fs_list)
        self.ps_mesh_static_original = ps.register_surface_mesh("Truss Static Original", Vs, Fs)
        self.ps_mesh_static_original.add_color_quantity("color", Cs, defined_on='faces')

        # Static optimized mesh
        Vs_list, Fs_list = self.truss_optimized.get_bar_geometry()
        Vs, Fs, Cs = self.combine_meshes(Vs_list, Fs_list)
        self.ps_mesh_static_optimized = ps.register_surface_mesh("Truss Static Optimized", Vs, Fs)
        self.ps_mesh_static_optimized.add_color_quantity("color", Cs, defined_on='faces')

        # Deformed mesh
        VD_list, FD_list = self.truss_original.get_deformed_bar_geometry(np.zeros(len(self.truss_original.nodes) * 6))
        Vd, Fd, Cd = self.combine_meshes(VD_list, FD_list)
        self.ps_mesh_deformed_original = ps.register_surface_mesh("Truss Deformed Original", Vd, Fd)
        self.ps_mesh_deformed_original.add_color_quantity("Displacement", Cd, defined_on='faces', enabled=True)
        self.ps_mesh_deformed_original.set_enabled(self.show_deformed)

        # Deformed optimized mesh
        VD_list, FD_list = self.truss_optimized.get_deformed_bar_geometry(np.zeros(len(self.truss_optimized.nodes) * 6))
        Vd, Fd, Cd = self.combine_meshes(VD_list, FD_list)
        self.ps_mesh_deformed_optimized = ps.register_surface_mesh("Truss Deformed Optimized", Vd, Fd)
        self.ps_mesh_deformed_optimized.add_color_quantity("Displacement", Cd, defined_on='faces', enabled=True)
        self.ps_mesh_deformed_optimized.set_enabled(self.show_deformed)

        # Force point visualization
        self.ps_force_pt = ps.register_point_cloud("force_point",
                                                   self.truss_optimized.nodes[self.force_node_id].reshape(1, 3),
                                                   enabled=self.show_deformed)
        self.ps_force_pt.set_color((1, 0, 0))
        self.ps_force_pt.set_radius(0.005)

        force_vec = np.array([self.force_x, self.force_y, self.force_z])
        self.ps_force_pt.add_vector_quantity("single point force",
                                             force_vec.reshape(1, 3),
                                             radius=0.005,
                                             enabled=self.show_deformed,
                                             color=(1.0, 0, 0))

    def callback(self):
        psim.PushItemWidth(150)

        if psim.Button("Compute truss deformation"):
            self.update_visualization_original()
            self.update_visualization_optimized()

        changed_view, self.show_deformed = psim.Checkbox("Show Deformed", self.show_deformed)
        if changed_view:
            self.ps_mesh_static_original.set_enabled(not self.show_deformed)
            self.ps_mesh_deformed_original.set_enabled(self.show_deformed)
            self.ps_mesh_static_optimized.set_enabled(not self.show_deformed)
            self.ps_mesh_deformed_optimized.set_enabled(self.show_deformed)
            self.ps_force_pt.set_enabled(self.show_deformed)

        changed_sw, self.consider_selfweight = psim.Checkbox("Self weight consideration", self.consider_selfweight)

        # Store current force values to check if they changed
        current_fx = self.force_x
        current_fy = self.force_y
        current_fz = self.force_z

        changed_fx, new_fx = psim.InputFloat("Force X", current_fx)
        if changed_fx:
            self.force_x = new_fx

        changed_fy, new_fy = psim.InputFloat("Force Y", current_fy)
        if changed_fy:
            self.force_y = new_fy

        changed_fz, new_fz = psim.InputFloat("Force Z", current_fz)
        if changed_fz:
            self.force_z = new_fz

        changed_vf, new_vf = psim.InputFloat("Volume Fraction", self.volume_fraction, format="%.2f")
        if changed_vf:
            self.volume_fraction = max(0.05, min(new_vf, 1.0))

        if psim.Button("Run Nonconvex Optimization"):
            optimizer = TrussOptimizerBinary(self.truss_optimized, self.get_force_vector(),
                                                volume_fraction=self.volume_fraction)
            self.optimized_densities = optimizer.solve_binary_optimization()
            if self.optimized_densities is not None:
                print(f"Optimization successful!")
                self.truss_optimized.update_bars_with_weight(self.optimized_densities)
                self.update_static_visualization_optimized()
            else:
                print("Optimization failed")

        psim.PopItemWidth()

    def get_force_vector(self):
        """Get global force vector including self-weight if enabled"""
        n_dofs_full = len(self.truss_optimized.nodes) * 6
        force_vector = np.zeros(n_dofs_full)

        # Add point forces
        force_vector[self.force_dof_x] = self.force_x
        force_vector[self.force_dof_y] = self.force_y
        force_vector[self.force_dof_z] = self.force_z

        # Consider self-weight if enabled
        if self.consider_selfweight:
            design_edges_set = self.truss_optimized.design_edges_set

            # Add self-weight for designed edges only
            for beam_id in range(len(self.truss_optimized.bars_elastic)):
                edge = self.truss_optimized.edges[beam_id]
                edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))

                if edge_norm in design_edges_set:
                    # Get self-weight load in global coordinates
                    load = self.truss_optimized.bars_elastic[beam_id].create_global_self_weight()

                    # Assembly the load into global force vector
                    for j in range(len(self.truss_optimized.edge_dof_indices[beam_id])):
                        dof = self.truss_optimized.edge_dof_indices[beam_id][j]
                        force_vector[dof] += load[j]

        return force_vector

    def show(self):
        ps.set_user_callback(self.callback)
        ps.show()


def main():
    # Create truss structure
    nx, ny, nz = 3, 3, 1

    # Define fixed nodes (left side)
    fixed_nodes = []
    for k in range(nz):
        for j in range(ny):
            node_id = 0 + j * nx + k * nx * ny
            fixed_nodes.append(node_id)

    # Define design edges
    design_edges = []

    # x-direction bars
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                node1 = i + j * nx + k * nx * ny
                node2 = (i + 1) + j * nx + k * nx * ny
                design_edges.append((node1, node2))

    # y-direction bars
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx):
                node1 = i + j * nx + k * nx * ny
                node2 = i + (j + 1) * nx + k * nx * ny
                design_edges.append((node1, node2))

    # z-layers
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                node1 = i + j * nx + k * nx * ny
                node2 = i + j * nx + (k + 1) * nx * ny
                design_edges.append((node1, node2))

    # Diagonal edges in xy planes
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx - 1):
                node1 = i + j * nx + k * nx * ny
                node2 = (i + 1) + (j + 1) * nx + k * nx * ny
                design_edges.append((node1, node2))

                node1 = (i + 1) + j * nx + k * nx * ny
                node2 = i + (j + 1) * nx + k * nx * ny
                design_edges.append((node1, node2))

    # Diagonal edges in xz planes
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx - 1):
                node1 = i + j * nx + k * nx * ny
                node2 = (i + 1) + j * nx + (k + 1) * nx * ny
                design_edges.append((node1, node2))

                node1 = (i + 1) + j * nx + k * nx * ny
                node2 = i + j * nx + (k + 1) * nx * ny
                design_edges.append((node1, node2))

    # Diagonal edges in yz planes
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx):
                node1 = i + j * nx + k * nx * ny
                node2 = i + (j + 1) * nx + (k + 1) * nx * ny
                design_edges.append((node1, node2))

                node1 = i + (j + 1) * nx + k * nx * ny
                node2 = i + j * nx + (k + 1) * nx * ny
                design_edges.append((node1, node2))

    # Create truss structures
    truss_original = TrussStructure(nx, ny, nz, fixed_nodes, design_edges)
    truss_optimized = TrussStructure(nx, ny, nz, fixed_nodes, design_edges)

    # # Define force application point (top-right corner)
    # force_node_id = nx * ny * (nz - 1) + nx - 1
    # force_dof_x = 6 * force_node_id + 0
    # force_dof_y = 6 * force_node_id + 1
    # force_dof_z = 6 * force_node_id + 2
    #
    # n_dofs_full = len(truss_original.nodes) * 6
    # force_vector = np.zeros(n_dofs_full)
    # force_vector[force_dof_y] = -1.0  # Apply force in y-direction

    # Create visualizer and start UI
    visualizer = TrussVisualizer(truss_original, truss_optimized)
    visualizer.setup_visualization()
    visualizer.show()

if __name__ == "__main__":
    main()

