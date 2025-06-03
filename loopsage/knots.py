from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import dblquad
import topoly as tp
import numpy as np
from tqdm import tqdm
from .utils import *
import os

################ Initial Knotted Structure Creation ######################

def resample_curve(points, N):
    deltas = np.diff(points, axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    cumulative = np.insert(np.cumsum(dists), 0, 0)
    uniform_samples = np.linspace(0, cumulative[-1], N)
    interp = interp1d(cumulative, points, axis=0)
    return interp(uniform_samples)

def trefoil_knot(N=200, scale=0.5, offset=(0, 0, 0), rotation=None):
    dense_N = 5000
    t = np.linspace(0, 2 * np.pi, dense_N)
    x = scale * (np.sin(t) + 2 * np.sin(2 * t)) + offset[0]
    y = scale * (np.cos(t) - 2 * np.cos(2 * t)) + offset[1]
    z = scale * (-np.sin(3 * t)) + offset[2]
    
    V = np.vstack((x, y, z)).T
    if rotation is not None:
        V = np.dot(V, rotation)
    return resample_curve(V, N)

def cinquefoil_knot(N=200, scale=0.5, offset=(0, 0, 0), rotation=None):
    dense_N = 5000
    t = np.linspace(0, 2 * np.pi, dense_N)
    x = scale * (np.sin(2*t) + 2*np.sin(3*t))
    y = scale * (np.cos(2*t) - 2*np.cos(3*t))
    z = scale * (-np.sin(5*t))
    
    V = np.vstack((x, y, z)).T
    if rotation is not None:
        V = np.dot(V, rotation)
    return resample_curve(V + np.array(offset), N)

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

def smooth_spline_linkage(start, start_dir, end, end_dir, N=100):
    """Generate a smooth cubic spline from `start` to `end` with given directional derivatives."""
    t = np.array([0, 1])
    x = np.array([start[0], end[0]])
    y = np.array([start[1], end[1]])
    z = np.array([start[2], end[2]])

    # Use directional derivative constraints
    cs_x = CubicSpline(t, x, bc_type=((1, start_dir[0]), (1, end_dir[0])))
    cs_y = CubicSpline(t, y, bc_type=((1, start_dir[1]), (1, end_dir[1])))
    cs_z = CubicSpline(t, z, bc_type=((1, start_dir[2]), (1, end_dir[2])))

    tt = np.linspace(0, 1, N)
    return np.vstack((cs_x(tt), cs_y(tt), cs_z(tt))).T

def random_unit_vector():
    """Uniformly sample a unit vector on the 3D sphere."""
    phi = np.random.uniform(0, 2*np.pi)
    costheta = np.random.uniform(-0.1, 0.1)
    sintheta = np.sqrt(1 - costheta**2)
    return np.array([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta])

def generate_knotted_structure(N=1000, num_knots=5):
    """Generate a structure with `num_knots` along a random walk, total `N` points."""
    if num_knots >= N:
        raise ValueError("N must be much larger than num_knots")

    points_per_knot = N // num_knots
    structure = []
    position = np.zeros(3)
    radius_buffer = 2.0  # Step size for knot placement

    for _ in range(num_knots):
        # Move to next knot position
        direction = random_unit_vector()
        position = position + direction * radius_buffer

        # Random rotation and scale
        rotation = random_rotation_matrix()
        scale = np.random.uniform(0.6, 1.0)

        # Generate knot
        knot_type = np.random.choice(["trefoil", "figure_eight"])
        if knot_type == "trefoil":
            knot = trefoil_knot(N=points_per_knot, scale=scale, offset=position, rotation=rotation)
        else:
            knot = cinquefoil_knot(N=points_per_knot, scale=scale, offset=position, rotation=rotation)

        structure.append(knot)

    curve = np.vstack(structure)
    return resample_curve(curve, N) if len(curve) != N else curve

############## Identification of Knots and Links ######################

def smooth_knot_spline(V, num_interp=200, closed=True):
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

def calculate_linking_number(V,ms,ns):
    links = list()
    for i in range(len(ms)):
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

def link_number_ensemble(path, step=1, mode='MD', viz=False):
    nlinks = list()
    Ms = np.load(path+'/other/Ms.npy')
    Ns = np.load(path+'/other/Ns.npy')
    
    # Automatically determine the number of ensembles
    ensemble_files = [f for f in os.listdir(path+'/ensemble') if f.startswith('MDLE_') and f.endswith('.cif')]
    max_index = max(int(f.split('_')[1].split('.')[0]) for f in ensemble_files)
    
    print('\nCalculating linking number of structures....')
    for i in tqdm(range(step, max_index + 1, step)):
        V = get_coordinates_cif(path+f'/ensemble/{mode}LE_{i}.cif')
        ms, ns = Ms[:, i-1], Ns[:, i-1]
        N_links, links = calculate_linking_number(V, ms, ns)
        nlinks.append(N_links)
    print('Done!')
    
    avg_link_number  = np.mean(nlinks)
    std_link_number  = np.std(nlinks)
    if viz:
        plt.figure()
        plt.plot(np.arange(0, len(nlinks)*step, step), nlinks)
        plt.xlabel('Step')
        plt.ylabel('Number of links')
        plt.savefig(path+'/plots/links.png')
        plt.close()

        plt.figure()
        plt.hist(nlinks, bins=20, alpha=0.75, color='blue', edgecolor='black')
        plt.xlabel('Number of links')
        plt.ylabel('Frequency')
        plt.title('Histogram of Number of Links')
        plt.savefig(path+'/plots/links_histogram.png')
        plt.close()
    np.save(path+'/other/links.npy', nlinks)
    print(f'Average number of links: {avg_link_number:.2f} ± {std_link_number:.2f}')
    return avg_link_number, std_link_number, nlinks