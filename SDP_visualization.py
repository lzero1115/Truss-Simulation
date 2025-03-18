import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from mosek.fusion import *
from scipy import sparse
from src.Truss_lex import TrussStructure


class TrussOptimizer:
    def __init__(self, truss, force_vector):
        self.truss = truss
        self.force_vector = force_vector  # 6 * nodes

    def solve_optimization(self):
        """Solve truss optimization using standard SDP form in Mosek.Fusion."""
        n_bars = len(self.truss.bars)
        n_dofs_full = len(self.truss.nodes) * 6
        n_dofs = self.truss.total_dofs  # reduced dofs
        matrix_size = n_dofs + 1
        total_volume = self.truss.get_total_volume()
        V = 0.5 * total_volume  # volume constraint
        epsilon = 1e-6  # for numerical stability

        print(f"Full DOFs: {n_dofs_full}")
        print(f"Reduced DOFs: {n_dofs}")

        # Identify design variables (edges that can be optimized)
        design_edges_indices = []
        for i, edge in enumerate(self.truss.edges):
            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_norm in self.truss.design_edges_set:
                design_edges_indices.append(i)

        n_design_vars = len(design_edges_indices)
        print(f"Total bars: {n_bars}")
        print(f"Design bars: {n_design_vars}")

        # Pre-compute stiffness matrices
        K_actual_all = []
        K_virtual_all = []
        for b in range(n_bars):
            k_actual, k_virtual = self.truss.get_bar_stiffness_matrix(b)
            K_actual_all.append(k_actual.toarray())
            K_virtual_all.append(k_virtual.toarray())

        # Map force vector to reduced space
        force_mapped = np.zeros(n_dofs)
        for i in range(n_dofs_full):
            if i in self.truss.map_dof_entire2subset:
                mapped_idx = self.truss.map_dof_entire2subset[i]
                force_mapped[mapped_idx] = self.force_vector[i]

        # Calculate combined stiffness for fixed (non-design) edges
        K_fixed = np.zeros((n_dofs, n_dofs))
        for i, edge in enumerate(self.truss.edges):
            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_norm not in self.truss.design_edges_set:
                # For non-design edges, add their full contribution
                K_fixed += K_actual_all[i]
            else:
                # For design edges, add their minimal contribution
                K_fixed += K_virtual_all[i]

        with Model("Truss_SDP_Standard") as M:
            # Variables - only create for design edges
            X = M.variable("X", Domain.inPSDCone(matrix_size))
            rho = M.variable("rho", n_design_vars, Domain.inRange(epsilon, 1.0))

            # Volume constraint
            vol_terms = []
            for idx, bar_idx in enumerate(design_edges_indices):
                vol_i = self.truss.bars[bar_idx].section_.Ax_ * self.truss.bars[bar_idx].length_
                vol_terms.append(Expr.mul(vol_i, rho.index(idx)))

            M.constraint("volume", Expr.add(vol_terms), Domain.lessThan(V))

            # Force vector constraints
            for i in range(n_dofs):
                M.constraint(Expr.sub(X.index(0, i + 1), -force_mapped[i]), Domain.equalsTo(0.0))
                M.constraint(Expr.sub(X.index(i + 1, 0), -force_mapped[i]), Domain.equalsTo(0.0))

            # Stiffness matrix constraints
            for i in range(n_dofs):
                for j in range(i, n_dofs):  # symmetric
                    # Start with fixed part of the stiffness matrix
                    k_constant = K_fixed[i, j]

                    # Add variable part from design edges
                    k_expr_terms = []
                    for var_idx, bar_idx in enumerate(design_edges_indices):
                        k_diff = K_actual_all[bar_idx][i, j] - K_virtual_all[bar_idx][i, j]
                        if abs(k_diff) > 1e-12:
                            k_expr_terms.append(Expr.mul(k_diff, rho.index(var_idx)))

                    if k_expr_terms:
                        k_expr = Expr.add(k_constant, Expr.add(k_expr_terms))
                    else:
                        k_expr = Matrix.dense([[k_constant]])

                    M.constraint(Expr.sub(X.index(i + 1, j + 1), k_expr), Domain.equalsTo(0.0))
                    if i != j:
                        M.constraint(Expr.sub(X.index(j + 1, i + 1), k_expr), Domain.equalsTo(0.0))

            # Objective
            M.objective(ObjectiveSense.Minimize, X.index(0, 0))

            # Solver parameters
            M.setSolverParam("intpntCoTolPfeas", 1e-6)
            M.setSolverParam("intpntCoTolDfeas", 1e-6)
            M.setSolverParam("intpntCoTolRelGap", 1e-6)
            M.setSolverParam("intpntSolveForm", "dual")
            M.setSolverParam("numThreads", 8)
            M.setSolverParam("optimizer", "conic")

            try:
                print("Running optimization...")

                M.solve()
                status = M.getPrimalSolutionStatus()
                print(f"Solution status: {status}")

                if status in [SolutionStatus.Optimal, SolutionStatus.Feasible]:
                    # Extract optimized design variables
                    design_rho = rho.level()
                    opt_tau = X.index(0, 0).level()

                    # Map back to full bar array
                    opt_rho = np.ones(n_bars)  # Initialize with all bars present
                    for i, bar_idx in enumerate(design_edges_indices):
                        opt_rho[bar_idx] = design_rho[i]

                    print(f"Optimal tau: {opt_tau}")
                    print("Sample rho values:", design_rho[:5])
                    return opt_rho, opt_tau
                else:
                    print(f"Problem status: {M.getProblemStatus()}")
                    print(f"Primal status: {M.getPrimalSolutionStatus()}")
                    print(f"Dual status: {M.getDualSolutionStatus()}")
                    return None, None
            except Exception as e:
                print(f"Optimization error type: {type(e).__name__}")
                print(f"Optimization error: {str(e)}")
                print("Optimization failed with exception")
                return None, None

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

        # Initialize force application point
        self.force_node_id = self.truss_original.nx * self.truss_original.ny * (self.truss_original.nz - 1) + self.truss_original.nx - 1
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

        if psim.Button("Run Optimization"):
            optimizer = TrussOptimizer(self.truss_optimized, self.get_force_vector())
            self.optimized_densities, tau = optimizer.solve_optimization()
            if self.optimized_densities is not None:
                print(f"Optimization successful! Tau: {tau}")
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

    # Create the truss structure
    truss_original = TrussStructure(nx=nx, ny=ny, nz=nz, fixed_nodes=fixed_nodes, design_edges=design_edges)
    truss_optimized = TrussStructure(nx=nx, ny=ny, nz=nz, fixed_nodes=fixed_nodes, design_edges=design_edges)

    # Create visualizer and run
    visualizer = TrussVisualizer(truss_original, truss_optimized)
    visualizer.setup_visualization()
    visualizer.show()

if __name__ == "__main__":
    main()

