import topoly as tp
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import scipy.interpolate as si
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import dblquad
from scipy.interpolate import splprep, splev, CubicSpline
from .uts import *

def trivial_knot(N=100, radius=1.0):
    """
    Generate a 3D trivial (unknot) structure as a simple circle.

    Parameters:
        N (int): Number of points in the loop.
        radius (float): Radius of the circular unknot.

    Returns:
        numpy.ndarray: Nx3 array representing the 3D coordinates of the unknot.
    """
    t = np.linspace(0, 2 * np.pi, N)  # Parameter along the circle
    x = radius * np.cos(t)  # Circle in the xy-plane
    y = radius * np.sin(t)
    z = np.zeros_like(t)  # No variation in z (flat circle)

    return np.vstack((x, y, z)).T  # Return as Nx3 matrix

def toroidal_knot(p, q, num_points=500, R=2, r=1):
    """
    Generate a toroidal (p, q) knot.
    
    Parameters:
        p (int): Number of turns around the torus' central axis.
        q (int): Number of turns around the torus' inner axis.
        num_points (int): Number of points to plot.
        R (float): Major radius of the torus (distance from center to torus tube).
        r (float): Minor radius of the torus tube.
    
    Returns:
        np.ndarray: Array of shape (num_points, 3) representing 3D coordinates of the knot.
    """
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Parametric equations for the toroidal knot
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    V = np.column_stack((x, y, z))
    V = np.vstack((V,[x[0],y[0],z[0]]))
    return V

def trefoil_knot(N=200, scale=0.8, offset=(0, 0, 0), rotation=None):
    """Generate a 3D Trefoil Knot with transformations."""
    t = np.linspace(0, 2 * np.pi-np.pi/36, N)
    x = scale * (np.sin(t) + 2 * np.sin(2 * t)) + offset[0]
    y = scale * (np.cos(t) - 2 * np.cos(2 * t)) + offset[1]
    z = scale * (-np.sin(3 * t)) + offset[2]
    
    V = np.vstack((x, y, z)).T

    if rotation is not None:
        V = np.dot(V, rotation)  # Apply rotation matrix

    return V

def cinquefoil_knot(N=300, scale=0.8, offset=(0, 0, 0), rotation=None):
    """Generate a 3D Cinquefoil (5_1) Knot with transformations."""
    t = np.linspace(0, 2 * np.pi, N)
    x = scale * (np.sin(2*t) + 2*np.sin(3*t))
    y = scale * (np.cos(2*t) - 2*np.cos(3*t))
    z = scale * (-np.sin(5*t))

    V = np.vstack((x, y, z)).T
    if rotation is not None:
        V = np.dot(V, rotation)  # Apply rotation
    return V + np.array(offset)  # Apply position offset

def random_rotation_matrix():
    """Generate a random 3D rotation matrix."""
    theta, phi, z = np.random.uniform(0, 2*np.pi-np.pi/36, 3)
    r = np.sqrt(z)
    V = np.array([
        [np.cos(theta) * np.cos(phi), -np.sin(theta), np.cos(theta) * np.sin(phi)],
        [np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    return V

def smooth_linkage(start, end, N=50):
    """Generate a smooth linking curve between two knots."""
    t = np.linspace(0, 1, N)
    x = (1 - t) * start[0] + t * end[0]
    y = (1 - t) * start[1] + t * end[1]
    z = (1 - t) * start[2] + t * end[2]
    return np.vstack((x, y, z)).T

def generate_knotted_structure(num_knots=5, N_per_knot=300):
    """Generate a large 3D knotted structure by linking multiple knots."""
    knots = []
    previous_end = None

    for i in range(num_knots):
        knot_type = np.random.choice(["trefoil", "figure_eight"])
        offset = previous_end + np.random.uniform(-3.0, 3.0, size=3) if previous_end is not None else np.random.uniform(-5, 5, size=3)
        scale = np.random.uniform(0.5, 1.0)  # Vary knot sizes
        rotation = random_rotation_matrix()
        
        if knot_type == "trefoil":
            knot = trefoil_knot(N=N_per_knot, scale=scale, offset=offset, rotation=rotation)
        else:
            knot = cinquefoil_knot(N=N_per_knot, scale=scale, offset=offset, rotation=rotation)
        
        if previous_end is not None:
            linkage = smooth_linkage(previous_end, knot[0], N=50)  # Smooth connection
            knots.append(linkage)

        knots.append(knot)
        previous_end = knot[-1]

    return np.vstack(knots)

def smooth_knot_spline(V, num_interp=100, closed=True):
    """
    Smoothly interpolates a 3D knot using a cubic spline.

    Parameters:
        V (numpy.ndarray): Nx3 array representing the 3D knot.
        num_interp (int): Number of points in the interpolated knot.
        closed (bool): Whether to enforce periodic boundary conditions for closed knots.

    Returns:
        numpy.ndarray: Smoothed Nx3 array of the interpolated knot.
    """
    if V.shape[1] != 3:
        raise ValueError("Input array V must have shape (N, 3)")
    
    if len(V) < 4:
        raise ValueError("At least 4 points are required for spline fitting.")

    # Define the parameter t along the curve
    t = np.linspace(0, 1, len(V))
    
    # Interpolation points
    t_new = np.linspace(0, 1, num_interp)

    # Fit cubic splines separately for each coordinate
    cs_x = CubicSpline(t, V[:, 0], bc_type='periodic' if closed else 'not-a-knot')
    cs_y = CubicSpline(t, V[:, 1], bc_type='periodic' if closed else 'not-a-knot')
    cs_z = CubicSpline(t, V[:, 2], bc_type='periodic' if closed else 'not-a-knot')

    # Evaluate the splines
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)

    return np.vstack((x_new, y_new, z_new)).T

def link_number_ensemble(path,N_ens=1250,step=100):
    nlinks = list()
    Ms = np.load(path+'/other/Ms.npy')
    Ns = np.load(path+'/other/Ns.npy')
    for i in range(N_ens//step,N_ens,step):
        V = uts.get_coordinates_cif(path+f'/ensemble/MDLE_{i+1}.cif')
        ms, ns = Ms[:,i], Ns[:,i]
        N_links, links = calculate_linking_number(V,ms,ns)
        nlinks.append(N_links)
    return nlinks

def calculate_linking_number(V,ms,ns):
    links = list()
    for i in tqdm(range(len(ms))):
        for j in range(i+1,len(ms)):
            if (ns[i]-ms[i])>5 and (ns[j]-ms[j])>5 and (((ms[i]<ns[i]) and (ns[i]<ms[j])) or ((ms[j]<ns[j]) and (ns[j]<ms[i]))):
                loop1 = V[ms[i]:ns[i]]
                loop1 = np.vstack((loop1,loop1[0,:]))
                loop1 = smooth_knot_spline(loop1,2*len(loop1))
                l1 = [list(loop1[i]) for i in range(len(loop1))]
                loop2 = V[ms[j]:ns[j]]
                loop2 = np.vstack((loop2,loop2[0,:]))
                loop2 = smooth_knot_spline(loop2,2*len(loop2))
                l2 = [list(loop2[i]) for i in range(len(loop2))]
                links.append(tp.gln(l1,l2))
    links = np.array(links)
    links = links[links>0.5]
    N_links = len(links)
    
    return N_links, links