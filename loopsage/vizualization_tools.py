#pip install imageio[ffmpeg]
import pyvista as pv
import imageio_ffmpeg  # Ensures ffmpeg is used
import imageio
import numpy as np
from .utils import *

def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

def viz_structure(V, colors=None, r=0.3, cmap='coolwarm'):
    # Handle the case where the structure is visualized as a single chain
    polyline = polyline_from_points(V)
    polyline["scalars"] = np.arange(polyline.n_points)

    if colors is not None:
        colors = colors[:len(V)]
        color_values = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))  # Normalize colors
        polyline["colors"] = color_values  # Set colors as point scalars
        polymer = polyline.tube(radius=r)
        polymer.plot(smooth_shading=True, cmap=cmap, scalars="colors", show_scalar_bar=False)
    else:
        polymer = polyline.tube(radius=r)
        polymer.plot(smooth_shading=True, show_scalar_bar=False)

def quaternion_from_matrix(matrix):
    """Convert a rotation matrix to a quaternion."""
    m = matrix
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2  # S=4*qw
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2  # S=4*qx
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2  # S=4*qy
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2  # S=4*qz
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw])

def rotation_matrix_from_quaternion(q):
    """Convert a quaternion to a rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def rmsd_quaternion(A, B):
    """Calculate the optimal rotation matrix to align A with B using quaternion RMSD."""
    assert A.shape == B.shape
    # Center the points
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)

    # Compute the covariance matrix
    H = np.dot(A_centered.T, B_centered)

    # Singular value decomposition
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)  # Optimal rotation matrix

    # Convert the rotation matrix to a quaternion
    q = quaternion_from_matrix(R)
    
    return R

def apply_rotation(points, rotation_matrix):
    """Apply rotation to the points."""
    return np.dot(points, rotation_matrix.T)

def create_structure_animation(V_list, filename='animation.gif', nframes=10, radius=0.2):
    pv.set_plot_theme("document")
    
    p = pv.Plotter()
    p.open_gif(filename)  # Open the GIF file here

    # Loop through the structures in the list
    for j in tqdm(range(len(V_list) - 1)):
        V1 = V_list[j]
        V2 = V_list[j + 1]
        
        # Compute the optimal rotation matrix
        R = rmsd_quaternion(V1, V2)
        
        # Calculate half the length
        half_len_V1 = len(V1) // 2
        half_len_V2 = len(V2) // 2
        
        # Loop through frames for interpolation
        for i in range(nframes):
            # Interpolating points between V1 and V2
            t = i / (nframes - 1)  # Normalized time
            
            # Interpolated points for chain 1 (first half)
            interpolated_chain1 = (1 - t) * V1[:half_len_V1] + t * V2[:half_len_V2]
            # Interpolated points for chain 2 (second half)
            interpolated_chain2 = (1 - t) * V1[half_len_V1:] + t * V2[half_len_V2:]
            
            # Apply rotation to both chains
            aligned_chain1 = apply_rotation(interpolated_chain1, R)
            aligned_chain2 = apply_rotation(interpolated_chain2, R)
            
            # Create polylines and tubes for both chains
            polyline1 = polyline_from_points(aligned_chain1)
            polyline2 = polyline_from_points(aligned_chain2)
            
            polymer1 = polyline1.tube(radius=radius)
            polymer2 = polyline2.tube(radius=radius)
            
            # Clear the previous mesh and add the new ones
            p.clear()
            p.add_mesh(polymer1, color='blue', smooth_shading=True, opacity=0.6)  # Chain 1
            p.add_mesh(polymer2, color='red', smooth_shading=True, opacity=0.6)    # Chain 2
            p.add_text(f"Structure {j}", font_size=16)
            
            # Write each frame to the GIF
            p.write_frame()
    print('Video is done! Give me some time to save the file now....')
    # Close and save the GIF
    p.close()
    print(f"Animation saved as {filename}")
    
def interactive_plot(ensemble_path):
    cifs = list_files_in_directory(ensemble_path)
    
    Vs = list()
    for cif in cifs:
        V = get_coordinates_cif(ensemble_path+'/'+cif)
        Vs.append(V)

    create_structure_animation(Vs)