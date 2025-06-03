import numpy as np
from tqdm import tqdm

def dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Mierzy dystans w przestrzeni R^3"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5  # faster than np.linalg.norm

def random_versor() -> np.ndarray:
    """Losuje wersor"""
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = np.random.uniform(-1, 1)
    d = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return np.array([x / d, y / d, z / d])

def self_avoiding_random_walk(n: int, step: float = 1.0, bead_radius: float = 0.5, epsilon: float = 0.001, two_dimensions=False) -> np.ndarray:
    potential_new_step = [0, 0, 0]
    while True:
        points = [np.array([0, 0, 0])]
        for _ in tqdm(range(n - 1)):
            step_is_ok = False
            trials = 0
            while not step_is_ok and trials < 1000:
                potential_new_step = points[-1] + step * random_versor()
                if two_dimensions:
                    potential_new_step[2] = 0
                for j in points:
                    d = dist(j, potential_new_step)
                    if d < 2 * bead_radius - epsilon:
                        trials += 1
                        break
                else:
                    step_is_ok = True
            points.append(potential_new_step)
        points = np.array(points)
        return points

def polymer_circle(n: int, z_stretch: float = 1.0, radius: float = 5.0) -> np.ndarray:
    points = []
    angle_increment = 360 / float(n)
    radius = 1 / (2 * np.sin(np.radians(angle_increment) / 2.)) if radius==None else radius
    z_stretch = z_stretch / n
    z = 0
    for i in range(n):
        x = radius * np.cos(angle_increment * i * np.pi / 180)
        y = radius * np.sin(angle_increment * i * np.pi / 180)
        if z_stretch != 0:
            z += z_stretch
        points.append((x, y, z))
    points = np.array(points)
    return points

def helix_structure(N_beads, radius=1, pitch=2):
    theta = np.linspace(0, 4 * np.pi, N_beads)  # 2 full turns
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(0, pitch * N_beads, N_beads)
    V = np.column_stack((x, y, z))
    return V

def spiral_structure(N_beads, initial_radius=1, pitch=1, growth_factor=0.05):
    theta = np.linspace(0, 4 * np.pi, N_beads)
    radius = initial_radius + growth_factor * np.arange(N_beads)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(0, pitch * N_beads, N_beads)
    V = np.column_stack((x, y, z))
    return V

def sphere_surface_structure(N_beads, radius=1):
    phi = np.random.uniform(0, 2 * np.pi, N_beads)
    costheta = np.random.uniform(-1, 1, N_beads)
    u = np.random.uniform(0, 1, N_beads)
    
    theta = np.arccos(costheta)
    r = radius * u ** (1/3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    V = np.column_stack((x, y, z))
    return V

def confined_random_walk(N_beads, box_size=5):
    V = np.zeros((N_beads, 3))
    for i in range(1, N_beads):
        step = np.random.choice([-1, 1], size=3)  # Random step in x, y, z
        V[i] = V[i-1] + step
        # Keep the points within a confined box (folding in)
        V[i] = np.clip(V[i], -box_size, box_size)
    return V

def trefoil_knot_structure(N_beads, scale=5):
    t = np.linspace(0, 2 * np.pi, N_beads)
    x = scale * (np.sin(t) + 2 * np.sin(2 * t))
    y = scale * (np.cos(t) - 2 * np.cos(2 * t))
    z = -scale * np.sin(3 * t)
    
    V = np.column_stack((x, y, z))
    return V

def random_walk_structure(N_beads, step_size=1):
    # Initialize the structure array
    V = np.zeros((N_beads, 3))
    
    # Loop over each bead, starting from the second one
    for i in range(1, N_beads):
        # Randomly pick a direction for each step
        step_direction = np.random.normal(size=3)
        step_direction /= np.linalg.norm(step_direction)  # Normalize to make it unit length
        
        # Move the current bead from the last bead by a fixed step size
        V[i] = V[i-1] + step_size * step_direction
    
    return V

def compute_init_struct(N_beads,mode='rw'):
    match mode:
        case 'rw':
            return random_walk_structure(N_beads)
        case 'confined_rw':
            return confined_random_walk(N_beads)
        case 'self_avoiding_rw':
            return self_avoiding_random_walk(N_beads)
        case 'circle':
            return polymer_circle(N_beads)
        case 'helix':
            return helix_structure(N_beads)
        case 'spiral':
            return spiral_structure(N_beads)
        case 'sphere':
            return sphere_surface_structure(N_beads)
        case _:
            return IndentationError('Invalid option for initial structure.')