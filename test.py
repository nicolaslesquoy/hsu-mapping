import numpy as np
from scipy.optimize import minimize

CENTER = (0.5, 0.5)  # Center for the test functions

def sphere(x: tuple) -> float:
    """
    This function computes the 'sphere' test function with centering (no rescaling needed).
    """
    return sum([(x[i] - CENTER[i]) ** 2 for i in range(len(x))])

def drop_wave(x: tuple) -> float:
    """
    This function computes the 'drop wave' test function with centering and rescaling.
    """
    scale = 1.5  # Arbitrary scaling factor
    return 1 - (
        1
        + np.cos(
            12
            * np.sqrt(sum([(scale * (x[i] - CENTER[i])) ** 2 for i in range(len(x))]))
        )
    ) / (0.5 * sum([(scale * (x[i] - CENTER[i])) ** 2 for i in range(len(x))]) + 2)

def rastrigin_mod(x: tuple) -> float:
    """
    This function computes the modified Rastrigin test function with centering and rescaling.
    Constants were removed and a non-linear term was added to the function.
    """
    scale = 10
    PI = 3.14159
    return sum(
        [
            (
                (scale * (x[i] - CENTER[i])) ** 2
                - 10 * np.cos(2 * PI * scale * (x[i] - CENTER[i]))
            )
            for i in range(len(x))
        ]
    ) - (scale * (x[0] - CENTER[0])) * (scale * (x[1] - CENTER[1]))

def rbf(r: np.ndarray, eps: float) -> np.ndarray:
    """Vectorized Radial Basis Function (RBF) kernel."""
    return 1/np.sqrt(r**2 + eps**2)  # Multiquadric kernel

def generate_grid(n: int, xmax: float = 1.0, ymax: float = 1.0) -> np.ndarray:
    x = np.linspace(0, xmax, n)
    y = np.linspace(0, ymax, n)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    return np.stack((xv, yv), axis=-1)

def generate_matrix(grid: np.ndarray, eps: float) -> np.ndarray:
    """Generate a matrix of RBF kernel values between all grid points."""
    # Reshape grid to (n*n, 2) array of points
    points = grid.reshape(-1, 2)
    n_points = len(points)
    
    # Vectorized distance calculation
    # Reshape for broadcasting: (n_points, 1, 2) - (1, n_points, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # Apply RBF kernel to the distance matrix
    return rbf(dist_matrix, eps)

def generate_function_vector(grid: np.ndarray, func) -> np.ndarray:
    """Generate a vector of function values for the given grid."""
    # Reshape to apply function to each point
    points = grid.reshape(-1, 2)
    
    # Vectorize for built-in functions or use list comprehension for custom functions
    func_vector = np.array([func(tuple(point)) for point in points])
    
    return func_vector

def cost_function(eps: float, grid: np.ndarray, function):
    """Compute the cost function based on the RBF kernel.
    """
    matrix = generate_matrix(grid, eps)
    inv_matrix = np.linalg.inv(matrix)
    func_vector = generate_function_vector(grid, function)
    error_vector = (inv_matrix @ func_vector) / np.diag(matrix)
    return np.linalg.norm(error_vector)
    
grid = generate_grid(64)
start_eps = 0.1

# minimize the cost function to find the optimal epsilon
result = minimize(
    cost_function,
    x0=[start_eps],
    args=(grid, rastrigin_mod),  # grid and sphere are now passed as args
    method='Nelder-Mead'
)
print("Optimal epsilon:", result.x[0])

