import numpy as np
import json
from src.Truss_lex import TrussStructure


class TrussMISDPGenerator:
    def __init__(self, truss, force_vector, force_node_id, reg=1e-6, scaling=1e5):
        """
        Initialize the TrussMISDPGenerator.

        Args:
            truss: The truss structure.
            force_vector: The force vector.
            force_node_id: The ID of the node where force is applied.
            reg: The regularization parameter to be added both per-bar and globally.
            scaling: Scaling factor for force, stiffness, and volume.
        """
        self.truss = truss
        self.force_vector = force_vector
        self.force_node_id = force_node_id
        self.reg = reg  # single regularization term
        self.scaling = scaling  # scaling factor

    def generate_sdpa_file(self, output_filename, volume_fraction=0.5, min_reg=1e-5):
        """
        Generate an SDPA (.dat-s) file for SCIP-SDP with improved numerical stability.
        For each design bar, adjust K_actual so that
            K_diff = K_actual - K_virtual
        is PSD (by adding reg to the diagonal of K_actual if needed).
        Then, assemble K_fixed (non-design edges: full K_actual; design edges: K_virtual),
        and add the same global reg.
        """
        # Increase regularization for better numerical stability
        self.reg = max(self.reg, min_reg)

        n_bars = len(self.truss.bars)
        n_dofs_full = len(self.truss.nodes) * 6
        n_dofs = self.truss.total_dofs
        matrix_size = n_dofs + 1
        total_volume = self.truss.get_total_volume()
        V = volume_fraction * total_volume

        # Identify design variables (edges that can be optimized)
        design_vars = []
        for i, edge in enumerate(self.truss.edges):
            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_norm in self.truss.design_edges_set:
                design_vars.append(i)
        n_vars = 1 + len(design_vars)

        print(f"Generating SCIP-SDP problem:")
        print(f"- Total edges: {n_bars}")
        print(f"- Design edges: {len(design_vars)}")
        print(f"- Variables: {n_vars} (1 tau + {len(design_vars)} design vars)")
        print(f"- Matrix size: {matrix_size}x{matrix_size}")
        print(f"- Scaling factor: {self.scaling}")
        print(f"- Regularization term (reg): {self.reg}")

        # Pre-compute stiffness matrices for each bar.
        # For design bars, adjust K_actual if needed so that (K_actual - K_virtual) is PSD.
        K_actual_all = []
        K_virtual_all = []
        for b in range(n_bars):
            k_actual, k_virtual = self.truss.get_bar_stiffness_matrix(b)
            k_actual_arr = k_actual.toarray()
            k_virtual_arr = k_virtual.toarray()
            edge = self.truss.edges[b]
            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_norm in self.truss.design_edges_set:
                diff = k_actual_arr - k_virtual_arr
                diff = 0.5 * (diff + diff.T)  # Ensure symmetry
                min_eig_diff = np.linalg.eigvalsh(diff).min()
                print(f"Design Bar {b}: initial min eigenvalue of (K_actual - K_virtual) = {min_eig_diff:.3e}")

                # Apply stronger regularization if needed
                if min_eig_diff < self.reg:
                    delta = self.reg - min_eig_diff + 1e-7  # Add a small buffer
                    print(f"  -> Adjusting Bar {b}: adding delta = {delta:.3e} to K_actual")
                    k_actual_arr += delta * np.eye(k_actual_arr.shape[0])
                    diff = k_actual_arr - k_virtual_arr
                    diff = 0.5 * (diff + diff.T)  # Ensure symmetry again
                    min_eig_diff = np.linalg.eigvalsh(diff).min()
                    print(f"  -> New min eigenvalue of (K_actual - K_virtual) = {min_eig_diff:.3e}")
            K_actual_all.append(k_actual_arr)
            K_virtual_all.append(k_virtual_arr)

        # Map force vector to reduced space
        force_mapped = np.zeros(n_dofs)
        for i in range(n_dofs_full):
            if i in self.truss.map_dof_entire2subset:
                mapped_idx = self.truss.map_dof_entire2subset[i]
                force_mapped[mapped_idx] = self.force_vector[i]
        force_mapped *= self.scaling

        # Assemble K_fixed: for non-design edges use full K_actual; for design edges use K_virtual.
        K_fixed = np.zeros((n_dofs, n_dofs))
        for i, edge in enumerate(self.truss.edges):
            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_norm not in self.truss.design_edges_set:
                K_fixed += K_actual_all[i]
            else:
                K_fixed += K_virtual_all[i]
        K_fixed *= self.scaling

        # Check eigenvalues of K_fixed before regularization
        min_eig_fixed = np.linalg.eigvalsh(K_fixed).min()
        print(f"Minimal eigenvalue of K_fixed before regularization: {min_eig_fixed:.3e}")

        # Add additional regularization if needed to ensure positive definiteness
        if min_eig_fixed < self.reg:
            additional_reg = self.reg - min_eig_fixed + 1e-7
            print(f"Adding additional regularization: {additional_reg:.3e} to ensure positive definiteness")
            K_fixed += additional_reg * np.eye(n_dofs)
        else:
            # Always add some regularization for numerical stability
            K_fixed += self.reg * np.eye(n_dofs)

        # Ensure perfect symmetry
        K_fixed = 0.5 * (K_fixed + K_fixed.T)

        # Diagnostic: Check eigenvalues of K_fixed after regularization.
        min_eig_fixed = np.linalg.eigvalsh(K_fixed).min()
        print(f"Minimal eigenvalue of K_fixed after regularization: {min_eig_fixed:.3e}")

        # Check condition number of K_fixed
        try:
            cond_num = np.linalg.cond(K_fixed)
            print(f"Condition number of K_fixed: {cond_num:.3e}")
            if cond_num > 1e15:
                print("  WARNING: K_fixed is very ill-conditioned!")
                print("  Consider increasing regularization parameter")
        except np.linalg.LinAlgError:
            print("  WARNING: Could not compute condition number for K_fixed!")

        # Verify all K_diff matrices remain PSD after scaling
        print("Checking K_diff matrices after scaling:")
        for var_idx, bar_idx in enumerate(design_vars):
            k_diff = self.scaling * (K_actual_all[bar_idx] - K_virtual_all[bar_idx])
            k_diff_symm = 0.5 * (k_diff + k_diff.T)  # Ensure symmetry
            min_eig = np.linalg.eigvalsh(k_diff_symm).min()
            print(f"  Bar {bar_idx} after scaling: min eigenvalue = {min_eig:.3e}")
            if min_eig < 0:
                print(f"    WARNING: K_diff is not PSD after scaling!")
                # Apply additional regularization if needed
                if min_eig < 0:
                    additional_reg = -min_eig + 1e-7
                    print(f"    Adding {additional_reg:.3e} to K_actual for bar {bar_idx}")
                    K_actual_all[bar_idx] += (additional_reg / self.scaling) * np.eye(K_actual_all[bar_idx].shape[0])
                    k_diff = self.scaling * (K_actual_all[bar_idx] - K_virtual_all[bar_idx])
                    k_diff_symm = 0.5 * (k_diff + k_diff.T)
                    min_eig = np.linalg.eigvalsh(k_diff_symm).min()
                    print(f"    -> New min eigenvalue = {min_eig:.3e}")

            # Check for near-zero K_diff matrices
            max_abs = np.max(np.abs(k_diff))
            if max_abs < 1e-6:
                print(f"    Warning: K_diff for bar {bar_idx} has max absolute value {max_abs:.3e}")

        # --- Diagnostic for LP block: Print sum of volume coefficients ---
        lp_sum = 0.0
        for i, bar_idx in enumerate(design_vars):
            vol_i = self.truss.bars[bar_idx].section_.Ax_ * self.truss.bars[bar_idx].length_
            vol_i_scaled = vol_i * self.scaling
            lp_sum += vol_i_scaled
        V_scaled = V * self.scaling
        print(f"LP block: Sum of volume coefficients = {lp_sum}, Volume upper bound = {V_scaled}")

        # Set threshold for sparse representation
        threshold = 1e-8
        force_mapped[np.abs(force_mapped) < threshold] = 0.0
        K_fixed[np.abs(K_fixed) < threshold] = 0.0

        # Identify bars connected to the force node
        connected_bars_indices = []
        for var_idx, bar_idx in enumerate(design_vars):
            edge = self.truss.edges[bar_idx]
            if self.force_node_id in edge:
                connected_bars_indices.append(var_idx)

        print(f"Found {len(connected_bars_indices)} design bars connected to force node {self.force_node_id}")

        # Calculate the LP block size (volume constraint + binary bounds + connectivity constraint)
        lp_block_size = 1 + 2 * len(design_vars) + 1  # Added 1 for connectivity constraint

        with open(output_filename, 'w') as f:
            f.write(f"{n_vars} = number of variables\n")
            f.write("2 = number of blocks\n")
            f.write(f"{matrix_size} -{lp_block_size} = blocksizes (negative sign for LP-block)\n")
            f.write("* objective function\n")
            coefs = ["1.0"] + ["0.0"] * len(design_vars)
            f.write(" ".join(coefs) + "\n")
            f.write("* constraints\n")

            # Tau entry in SDP block (position (1,1))
            f.write("1 1 1 1 1.0\n")

            # Force vector entries in SDP block (upper triangle only)
            for i in range(n_dofs):
                if abs(force_mapped[i]) > threshold:
                    # upper triangle
                    f.write(f"0 1 1 {i + 2} {force_mapped[i]}\n") # -A0 + sum(A_i)


            # Fixed stiffness entries (upper triangle)
            for i in range(n_dofs):
                for j in range(i, n_dofs):
                    if abs(K_fixed[i, j]) > threshold:
                        f.write(f"0 1 {i + 2} {j + 2} {K_fixed[i, j]}\n")

            # Variable stiffness entries for design edges (upper triangle)
            for var_idx, bar_idx in enumerate(design_vars):
                k_diff = self.scaling * (K_actual_all[bar_idx] - K_virtual_all[bar_idx])
                k_diff = 0.5 * (k_diff + k_diff.T)  # Ensure symmetry when writing
                k_diff[np.abs(k_diff) < threshold] = 0.0
                for i in range(n_dofs):
                    for j in range(i, n_dofs):
                        if abs(k_diff[i, j]) > threshold:
                            f.write(f"{var_idx + 2} 1 {i + 2} {j + 2} {k_diff[i, j]}\n")

            # Volume constraint: sum_i vol_i * x_i <= V_max
            lp_row = 1
            for i, bar_idx in enumerate(design_vars):
                vol_i = self.truss.bars[bar_idx].section_.Ax_ * self.truss.bars[bar_idx].length_
                vol_i_scaled = vol_i * self.scaling
                if abs(vol_i_scaled) > threshold:
                    f.write(f"{i + 2} 2 {lp_row} {lp_row} {-vol_i_scaled}\n")  # Negative coefficient for variable
            f.write(f"0 2 {lp_row} {lp_row} {-V_scaled}\n")  # Negative constant term

            # Add explicit bounds for binary variables
            for i in range(len(design_vars)):
                # Upper bound: ρᵢ ≤ 1 ---> -ρᵢ ≥ -1
                lp_row += 1
                f.write(f"{i + 2} 2 {lp_row} {lp_row} -1.0\n")
                f.write(f"0 2 {lp_row} {lp_row} -1.0\n")

                # Lower bound: ρᵢ ≥ 0
                lp_row += 1
                f.write(f"{i + 2} 2 {lp_row} {lp_row} 1.0\n")
                f.write(f"0 2 {lp_row} {lp_row} 0.0\n")

            # Add connectivity constraint: sum of bars connected to force node >= 1
            if connected_bars_indices:
                lp_row += 1
                for var_idx in connected_bars_indices:
                    f.write(f"{var_idx + 2} 2 {lp_row} {lp_row} 1.0\n")
                f.write(f"0 2 {lp_row} {lp_row} -1.0\n")
                print(f"Added connectivity constraint for force node {self.force_node_id}")
            else:
                print(f"WARNING: No design bars connected to force node {self.force_node_id}!")
                print(f"The problem will likely be infeasible!")

            # Mark variables as integer
            f.write("*INTEGER\n")
            for i in range(2, n_vars + 1):
                f.write(f"*{i}\n")

        print(f"SDPA file written to {output_filename}")

        var_mapping = {'tau_idx': 1, 'design_vars': design_vars, 'n_bars': n_bars}
        with open(output_filename + ".map", 'w') as f:
            json.dump(var_mapping, f)
        return var_mapping


def main():
    nx, ny, nz = 3, 3, 1
    fixed_nodes = []
    for k in range(nz):
        for j in range(ny):
            node_id = 0 + j * nx + k * nx * ny
            fixed_nodes.append(node_id)
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
    # z-direction bars
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

    truss = TrussStructure(nx=nx, ny=ny, nz=nz, fixed_nodes=fixed_nodes, design_edges=design_edges)
    force_vector = np.zeros(len(truss.nodes) * 6)
    force_node_id = truss.nx * truss.ny * (truss.nz - 1) + truss.nx - 1
    force_dof_y = 6 * force_node_id + 1
    force_vector[force_dof_y] = -1

    # Use higher regularization and appropriate scaling
    generator = TrussMISDPGenerator(truss, force_vector, force_node_id, reg=1e-5, scaling=1)
    generator.generate_sdpa_file("truss_binary.dat-s", volume_fraction=0.5, min_reg=1e-4)


if __name__ == "__main__":
    main()