import numpy as np
from typing import List, Tuple, Dict, Optional, Set

from mpmath import cplot
from scipy import sparse
from scipy.sparse import linalg
from .BarLinearElastic import BarLinearElastic
from .Bar import Bar
from .BarMaterial import BarMaterial
from .BarCrossSection import BarCrossSection, BarCrossSectionRound, CrossSectionType
from .CoordinateSystem import CoordinateSystem
from .jetmap import jetmap


class TrussStructureError(Exception):
    """Base exception class for TrussStructure errors"""
    pass


class InvalidNodeError(TrussStructureError):
    """Exception for invalid node configurations"""
    pass


class TrussStructure:
    def __init__(self, nx: int, ny: int, nz: int, fixed_nodes: List[int], design_edges: List[Tuple[int, int]]):
        # Input validation
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("Grid dimensions must be positive")
        total_nodes = nx * ny * nz
        if any(node >= total_nodes or node < 0 for node in fixed_nodes):
            raise InvalidNodeError("Fixed node indices must be within grid bounds")

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.fixed_nodes = fixed_nodes
        self.design_edges = design_edges
        self.design_edges_set = {(min(e[0], e[1]), max(e[0], e[1])) for e in design_edges}
        self.bar_discretization = 10

        #consider self weight
        self.consider_selfweight = False

        # Material properties
        self.E = 5e6  # Pa
        self.G = 46071428.57142857  # Pa
        self.rho = 58.0  # kg/m^3
        self.mu = self.E / (2 * self.G) - 1

        # Cross-sectional properties
        self.min_radius = 1e-6  # numerical stability virtual bar
        self.min_section = BarCrossSectionRound(radius=self.min_radius)
        self.design_area = 7.853981633974483e-05  # m^2
        self.design_radius = np.sqrt(self.design_area / np.pi)
        self.design_section = BarCrossSectionRound(radius=self.design_radius)

        # Initialize structures
        self.nodes: List[np.ndarray] = []
        self.edges: List[Tuple[int, int]] = []

        self.bars: List[Bar] = []
        self.bars_elastic: List[BarLinearElastic] = []
        self.vertex_dof_indices: List[List[int]] = []
        self.edge_dof_indices: List[List[int]] = []

        # Initialize the structure
        self.initialize_grid_structure()
        self.initialize_structure()

    def get_node_index(self, i: int, j: int, k: int) -> int:
        """Convert 3D grid coordinates to node index."""
        return i + j * self.nx + k * self.nx * self.ny

    def initialize_grid_structure(self):
        """Initialize nodes and edges for the grid structure."""
        # Create nodes
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    self.nodes.append(np.array([float(i), float(j), float(k)]))

        def add_edge(i1: int, j1: int, k1: int, i2: int, j2: int, k2: int):
            if (0 <= i1 < self.nx and 0 <= j1 < self.ny and 0 <= k1 < self.nz and
                    0 <= i2 < self.nx and 0 <= j2 < self.ny and 0 <= k2 < self.nz):
                n1 = self.get_node_index(i1, j1, k1)
                n2 = self.get_node_index(i2, j2, k2)
                if n1 != n2:
                    self.edges.append((min(n1, n2), max(n1, n2)))

        # Create base grid edges (only square frame)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    if i < self.nx - 1:
                        add_edge(i, j, k, i + 1, j, k)
                    if j < self.ny - 1:
                        add_edge(i, j, k, i, j + 1, k)
                    if k < self.nz - 1:
                        add_edge(i, j, k, i, j, k + 1)

        # Add design edges if not already present
        existing_edges_set = set(self.edges)
        for e in self.design_edges_set:
            if e not in existing_edges_set:
                self.edges.append(e)

    def initialize_structure(self):
        """Initialize bars and setup for simulation."""
        self.compute_bars()
        self.compute_bars_linear_elasticity()

    def compute_bars(self):
        """Create bars from edges and nodes."""

        self.bars.clear()

        for edge in self.edges:
            start_node = self.nodes[edge[0]]
            end_node = self.nodes[edge[1]]
            coord = CoordinateSystem(origin=start_node, zaxis=end_node - start_node)

            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            section = self.design_section if edge_norm in self.design_edges_set else self.min_section
            length = float(np.linalg.norm(end_node - start_node))
            bar = Bar(coord, length, section)
            self.bars.append(bar)

    def compute_bars_linear_elasticity(self):
        """Setup elastic bars and degrees of freedom for simulation."""
        self.bars_elastic.clear()
        self.vertex_dof_indices.clear()
        self.edge_dof_indices.clear()

        for i in range(len(self.nodes)):
            dofs = [i * 6 + j for j in range(6)]
            self.vertex_dof_indices.append(dofs)

        # each bar is assigned with two nodes
        for i, bar in enumerate(self.bars):
            material = BarMaterial(E=self.E, mu=self.mu, rho=self.rho, section=bar.section_)
            bar_elastic = BarLinearElastic(bar, material)
            self.bars_elastic.append(bar_elastic)

            edge_dof = []
            for vertex_id in self.edges[i]:
                edge_dof.extend(self.vertex_dof_indices[vertex_id])
            self.edge_dof_indices.append(edge_dof)

    # core code
    def _compute_dofs_mapping(self, map_entire2subset: Dict[int, int],
                              map_subset2entire: Dict[int, int]) -> int:
        """Compute DoF mapping excluding fixed nodes."""
        new_dof = 0
        for dof in range(6 * len(self.nodes)):
            node_id = dof // 6
            if node_id not in self.fixed_nodes:
                map_entire2subset[dof] = new_dof
                map_subset2entire[new_dof] = dof
                new_dof += 1
        return new_dof

    def _compute_stiff_matrix(self, total_dofs: int,
                              map_dof_entire2subset: Dict[int, int]) -> sparse.csr_matrix:
        """Compute global stiffness matrix."""
        triplet_list = []

        for i, elastic_bar in enumerate(self.bars_elastic):
            k_G = elastic_bar.create_global_stiffness_matrix()
            self._assemble_stiff_matrix(triplet_list, k_G, map_dof_entire2subset, i)

        if triplet_list:
            rows, cols, data = zip(*triplet_list)
            return sparse.csr_matrix((data, (rows, cols)), shape=(total_dofs, total_dofs))
        return sparse.csr_matrix((total_dofs, total_dofs))

    def _assemble_stiff_matrix(self, K_tri: List[Tuple[int, int, float]],
                               k_G: np.ndarray, map_dof_entire2subset: Dict[int, int],
                               edge_id: int):
        """Assemble element stiffness matrix into global stiffness matrix."""
        new_dofs = [] # degenerate the fixed nodes degrees
        for j in range(len(self.edge_dof_indices[edge_id])):
            old_dof = self.edge_dof_indices[edge_id][j]
            new_dofs.append(map_dof_entire2subset.get(old_dof, -1))

        for j in range(12):
            for k in range(12):
                if abs(float(k_G[j, k])) < 1e-12:
                    continue
                new_dof_j = new_dofs[j]
                new_dof_k = new_dofs[k]
                if new_dof_j != -1 and new_dof_k != -1:
                    K_tri.append((new_dof_j, new_dof_k, float(k_G[j, k])))

    def _compute_loads(self, tot_dofs: int, map_dof_entire2subset: Dict[int, int],
                       external_forces: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute global load vector including self-weight for designed edges and external forces."""
        F = np.zeros(tot_dofs)

        # Get the set of designed edges for efficient lookup
        if self.consider_selfweight:
            design_edges_set = self.design_edges_set

            # Add self-weight for designed edges only
            for beam_id in range(len(self.bars_elastic)):
                edge = self.edges[beam_id]
                edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))

                if edge_norm in design_edges_set:  # Only consider designed edges for self-weight
                    load = self.bars_elastic[beam_id].create_global_self_weight()
                    self._assembly_force(F, load, map_dof_entire2subset, beam_id)

        # Add external forces if provided
        if external_forces is not None:
            for i in range(external_forces.shape[0]):
                if i in map_dof_entire2subset:
                    F[map_dof_entire2subset[i]] += external_forces[i]

        return F

    def _assembly_force(self, F: np.ndarray, g: np.ndarray,
                        map_dof_entire2subset: Dict[int, int], edge_id: int):
        """Assemble element force vector into global force vector."""
        new_dofs = []
        for j in range(len(self.edge_dof_indices[edge_id])):
            old_dof = self.edge_dof_indices[edge_id][j]
            new_dofs.append(map_dof_entire2subset.get(old_dof, -1))

        for j, new_dof in enumerate(new_dofs):
            if new_dof != -1:
                F[new_dof] += g[j]

    def solve_elasticity(self, external_forces: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, str]:
        """Solve elasticity problem for the truss structure."""
        try:
            # Setup DoF mapping
            map_dof_entire2subset = {}
            map_dof_subset2entire = {}
            total_dofs = self._compute_dofs_mapping(map_dof_entire2subset, map_dof_subset2entire)

            # Compute stiffness matrix and loads
            K = self._compute_stiff_matrix(total_dofs, map_dof_entire2subset)
            F = self._compute_loads(total_dofs, map_dof_entire2subset, external_forces)

            # Solve system
            D = sparse.linalg.spsolve(K, F)

            # Map solution back to full system
            displacement = np.zeros(len(self.nodes) * 6)
            for i in range(D.shape[0]):
                old_dof = map_dof_subset2entire[i]
                displacement[old_dof] = D[i]

            return displacement, True, "Success"
        except Exception as e:
            return np.zeros(len(self.nodes) * 6), False, f"Failed to solve: {str(e)}"

    def get_bar_volumes(self) -> List[float]:
        """Calculate volumes of all bars."""
        volumes = []
        for bar in self.bars:
            volume = bar.length_ * bar.section_.Ax_
            volumes.append(volume)
        return volumes

    def get_total_volume(self) -> float:
        """Calculate total volume of the structure."""
        return sum(self.get_bar_volumes())

    def get_strain_energy(self, displacement: np.ndarray) -> float: # compliance
        """Calculate total strain energy of the structure."""
        energy = 0.0
        for i, elastic_bar in enumerate(self.bars_elastic):
            u = np.zeros(12)
            for k in range(2):
                node_id = self.edges[i][k]
                u[k * 6:(k + 1) * 6] = displacement[node_id * 6:(node_id + 1) * 6]
            f = elastic_bar.compute_internal_force(u)
            energy += 0.5 * f.dot(u)
        return energy

    def get_compliance(self, displacement: np.ndarray, external_forces: np.ndarray) -> float:
        """Calculate compliance (work done by external forces)."""
        return 0.5 * displacement.dot(external_forces)

    def interpolate_polynomial(self, D_local: np.ndarray, L: float) -> List[np.ndarray]:
        """Interpolate displacement polynomial."""
        #TODO: modification of corresponding sign later
        d_y = np.array([D_local[1], D_local[7], D_local[5], D_local[11]])
        #d_z = np.array([D_local[2], D_local[8], D_local[4], D_local[10]])
        d_z = np.array([D_local[2], D_local[8], -D_local[4], -D_local[10]])

        A = np.zeros((4, 4))
        u0, u6 = D_local[0], D_local[6] + L

        A[0] = [1.0, u0, u0 * u0, u0 * u0 * u0]
        A[1] = [1.0, u6, u6 * u6, u6 * u6 * u6]
        A[2] = [0.0, 1.0, 2 * u0, 3 * u0 * u0]
        A[3] = [0.0, 1.0, 2 * u6, 3 * u6 * u6]

        return [np.linalg.solve(A, d_y), np.linalg.solve(A, d_z)]

    def visualize_displacement(self, bar_id: int, displacement: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Visualize displacement for a single bar."""

        def poly_val(x: float, c: np.ndarray) -> float:
            return sum(c[i] * x ** i for i in range(len(c)))

        node_u, node_v = self.edges[bar_id]
        end_u = self.nodes[node_u]
        end_v = self.nodes[node_v]

        R3 = self.bars_elastic[bar_id].create_global_transformation_matrix()
        R = self.bars_elastic[bar_id].turn_diagblock(R3)

        D_global = np.zeros(12)
        D_global[:6] = displacement[node_u * 6:(node_u + 1) * 6]
        D_global[6:] = displacement[node_v * 6:(node_v + 1) * 6]
        D_local = R @ D_global # local

        L = self.bars_elastic[bar_id].length_
        u_poly = self.interpolate_polynomial(D_local, L)

        polylines = []
        distance = []

        for i in range(self.bar_discretization + 1):
            t = i / self.bar_discretization
            s = D_local[0] + t * (D_local[6] + L - D_local[0])
            v = poly_val(s, u_poly[0])
            w = poly_val(s, u_poly[1])
            d = np.array([s, v, w]) # local displacement

            inter_pt = end_u + R3.T @ d
            ori_pt = end_u + t * (end_v - end_u)

            polylines.append(inter_pt) # global deformed section position
            distance.append(float(np.linalg.norm(inter_pt - ori_pt)))

        return polylines, distance

    def visualize_displacement_all(self, displacement: np.ndarray) -> Tuple[List[List[np.ndarray]], List[List[float]]]:
        """Visualize displacement for design edges only."""
        # displacement is global variable
        segments_list = []
        deviation_list = []

        for i, edge in enumerate(self.edges):
            edge_normalized = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_normalized in self.design_edges_set:
                segments, deviations = self.visualize_displacement(i, displacement)
                segments_list.append(segments)
                deviation_list.append(deviations)

        return segments_list, deviation_list

    def get_bar_geometry(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return the static (undeformed) 3D mesh of each designed bar."""
        Vs, Fs = [], []
        for i, bar in enumerate(self.bars):
            edge_norm = (min(self.edges[i][0], self.edges[i][1]), max(self.edges[i][0], self.edges[i][1]))
            if edge_norm in self.design_edges_set:
                V, F = bar.get_mesh()
                Vs.append(V)
                Fs.append(F)
        return Vs, Fs

    def get_deformed_bar_geometry(self, displacement: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return the deformed 3D mesh of each designed bar."""
        Vs, Fs = [], []
        for bar_id, edge in enumerate(self.edges):
            edge_norm = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            if edge_norm in self.design_edges_set:
                segments, _ = self.visualize_displacement(bar_id, displacement)
                for i in range(self.bar_discretization):
                    end_u, end_v = segments[i], segments[i + 1]
                    cross_section = self.bars[bar_id].section_
                    bar = Bar.from_points(end_u, end_v, cross_section)
                    V, F = bar.get_mesh()
                    Vs.append(V)
                    Fs.append(F)
        return Vs, Fs

    def get_deformed_bar_displacement_colors(self, displacement: np.ndarray, max_disp: float) -> List[np.ndarray]:
        """Get colors based on deformation."""
        colors = []
        segments, deviations = self.visualize_displacement_all(displacement)

        for bar_id, bar_deviations in enumerate(deviations):
            edge_norm = (min(self.edges[bar_id][0], self.edges[bar_id][1]),
                         max(self.edges[bar_id][0], self.edges[bar_id][1]))
            if edge_norm in self.design_edges_set:
                for j in range(self.bar_discretization):
                    dev = (bar_deviations[j] + bar_deviations[j+1])/2.0
                    color = jetmap(dev, 0, max_disp)
                    colors.append(color)

        return colors



