import polyscope as ps
import polyscope.imgui as psim  # For custom UI components
import numpy as np
from typing import Optional, List

from torch.backends.cudnn import enabled

from src.Truss_lex import TrussStructure

# global UI variables
show_deformed = False
consider_selfweight = False
# single point applied force
force_x = 0.0
force_y = 0.0
force_z = 0.0
# global mesh variable
ps_mesh_static = None
ps_mesh_deformed = None
truss = None
c_id = None
ps_force_pt = None
ps_force_vec = None


def combine_meshes(V_list, F_list, disp: Optional[List[np.ndarray]]):
    total_vertices = np.sum([v.shape[0] for v in V_list])
    total_faces = np.sum([f.shape[0] for f in F_list])
    V = np.zeros((total_vertices,3))
    F = np.zeros((total_faces,3), dtype=int)
    C = np.zeros((total_faces,3)) # color
    v_offset = 0
    f_offset = 0
    for i in range(len(V_list)):
        v = V_list[i]
        f = F_list[i]
        V[v_offset:v_offset+v.shape[0]] = v
        F[f_offset:f_offset+f.shape[0]] = f + v_offset
        if disp is not None:
            C[f_offset:f_offset + f.shape[0]] = disp[i]
        else:
            C[f_offset:f_offset+f.shape[0]] = np.array([0.8,0.8,0.8]) # default gray color
        v_offset += v.shape[0]
        f_offset += f.shape[0]

    return V, F, C


def update():
    global ps_mesh_static, ps_mesh_deformed, truss, c_id, ps_force_pt, ps_force_vec, force_x, force_y, force_z
    tot_dof = len(truss.nodes)*6
    eforce = np.zeros(tot_dof)
    fx_dof = 6 * c_id + 0
    fy_dof = 6 * c_id + 1
    fz_dof = 6 * c_id + 2
    eforce[fx_dof] = force_x
    eforce[fy_dof] = force_y
    eforce[fz_dof] = force_z
    ps_force_vec = np.array([force_x,force_y,force_z])
    truss.consider_selfweight = consider_selfweight

    # Solve elasticity
    displacement, success, message = truss.solve_elasticity(eforce)

    if success:
        max_disp = np.max(np.abs(displacement))
        print(f"Elasticity solved successfully! Max Displacement: {max_disp}")

        disp_color = truss.get_deformed_bar_displacement_colors(displacement, max_disp)
        VD_list, FD_list = truss.get_deformed_bar_geometry(displacement)
        Vd, Fd, Cd = combine_meshes(VD_list, FD_list, disp_color)

        # Update deformed mesh
        ps_mesh_deformed.update_vertex_positions(Vd)
        ps_mesh_deformed.add_color_quantity("Displacement", Cd, defined_on='faces', enabled=True)
        ps_mesh_deformed.set_enabled(show_deformed)
        ps_mesh_static.set_enabled(not show_deformed)


        node_displacement = displacement[6 * c_id:6 * c_id + 3]
        force_point = truss.nodes[c_id] + node_displacement
        ps_force_pt.update_point_positions(force_point.reshape(1, 3))


    else:
        print("Elasticity solved failed:", message)

def callback():
    global show_deformed, consider_selfweight, force_x, force_y, force_z,ps_force_pt,ps_mesh_deformed,ps_force_vec,ps_mesh_static
    psim.PushItemWidth(150)
    # psim.TextUnformatted("Truss Visualization Control")
    # psim.Separator()
    help = False
    if (psim.Button("Compute truss deformation")):
        help = not help

    #scaled_vl = np.linalg.norm(ps_force_vec)

    changed_view, show_deformed = psim.Checkbox("Show Deformed", show_deformed)
    if changed_view:
        ps_mesh_static.set_enabled(not show_deformed)
        ps_mesh_deformed.set_enabled(show_deformed)
        ps_force_pt.set_enabled(show_deformed)
        ps_force_pt.add_vector_quantity("single point force", ps_force_vec.reshape(1, 3),
                                        radius=0.005, enabled=show_deformed,
                                        color=(1.0, 0, 0))

    changed_sw, consider_selfweight = psim.Checkbox("Self weight consideration", consider_selfweight)

    changed_fx, force_x = psim.InputFloat("Force X", force_x)
    changed_fy, force_y = psim.InputFloat("Force Y", force_y)
    changed_fz, force_z = psim.InputFloat("Force Z", force_z)

    if help:
        update()

    psim.PopItemWidth()



def main():
    global ps_mesh_static, ps_mesh_deformed, truss, c_id, ps_force_pt, ps_force_vec
    # create the truss structure

    nx, ny, nz = 3, 3, 1

    fixed_nodes = []
    for k in range(nz):
        for j in range(ny):
            node_id = 0 + j*nx + k*nx*ny # i = 0 for left side
            fixed_nodes.append(node_id)

    design_edges = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx-1): # x-direction

                node1 = i + j*nx + k*nx*ny
                node2 = (i+1) + j*nx + k*nx*ny
                design_edges.append((node1,node2))

            if j < ny-1: # y direction
                for i in range(nx):
                    
                    node1 = i + j * nx + k * nx * ny
                    node2 = i + (j+1) * nx + k * nx * ny
                    design_edges.append((node1,node2))

    # connection between layers (z-direction)
    for k in range(nz-1):
        for j in range(ny):
            for i in range(nx):
                node1 = i + j*nx + k*nx*ny
                node2 = i + j*nx + (k+1)*nx*ny
                design_edges.append((node1,node2))

    # Add diagonal bracing (optional)
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
    # for j in range(ny):
    #     for k in range(nz - 1):
    #         for i in range(nx - 1):
    #             node1 = i + j * nx + k * nx * ny
    #             node2 = (i + 1) + j * nx + (k + 1) * nx * ny
    #             design_edges.append((node1, node2))
    #
    #             node1 = (i + 1) + j * nx + k * nx * ny
    #             node2 = i + j * nx + (k + 1) * nx * ny
    #             design_edges.append((node1, node2))

    truss = TrussStructure(nx,ny,nz,fixed_nodes,design_edges)
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    Vs_list, Fs_list = truss.get_bar_geometry()
    Vs, Fs, Cs = combine_meshes(Vs_list, Fs_list, None)
    ps_mesh_static = ps.register_surface_mesh("Truss Static", Vs, Fs)
    ps_mesh_static.add_color_quantity("color", Cs, defined_on='faces')

    # single node control
    c_id = nx*ny*(nz-1) + nx - 1 # front bottom right node

    # Create initial deformed mesh
    VD_list, FD_list = truss.get_deformed_bar_geometry(np.zeros(len(truss.nodes) * 6))
    Vd, Fd, Cd = combine_meshes(VD_list, FD_list, None)
    ps_mesh_deformed = ps.register_surface_mesh("Truss Deformed", Vd, Fd)
    ps_mesh_deformed.add_color_quantity("Displacement", Cd, defined_on='faces',enabled =True)
    ps_mesh_deformed.set_enabled(show_deformed)
    ps_force_pt = ps.register_point_cloud("force_point", truss.nodes[c_id].reshape(1, 3),enabled=show_deformed)
    ps_force_vec = np.zeros(3)
    ps_force_pt.add_vector_quantity("single point force", ps_force_vec.reshape(1, 3),
                                    radius=0.005, enabled=show_deformed,
                                    color=(1.0, 0.8, 0.8))
    ps_force_pt.set_color((1, 0, 0))
    ps_force_pt.set_radius(0.005)
    ps_force_pt.set_enabled(show_deformed)


    ps.set_user_callback(callback)
    ps.show()

if __name__ == "__main__":
    main()

