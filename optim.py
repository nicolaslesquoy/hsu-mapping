from typing import Callable
from typing import Optional
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import shgo
from scipy.spatial.distance import cdist

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = """\\usepackage{lmodern}"""

CENTER = np.array([0.5, 0.5])  # Center of the domain used as center for RBFs
THRESHOLD = 1e-9  # Threshold for plotting

p = 1 / 2.54  # Conversion factor from inches to centimeters

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### TEST FUNCTIONS ###
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def ackley(x: tuple) -> float:
    """
    This function computes Ackley's test function with centering and rescaling.
    """
    scale = 5  # Arbitrary scaling factor
    PI = 3.14159
    return (
        -20
        * np.exp(
            -0.2
            * np.sqrt(
                1
                / len(x)
                * sum([(scale * (x[i] - CENTER[i])) ** 2 for i in range(len(x))])
            )
        )
        - np.exp(
            1
            / len(x)
            * sum(
                [np.cos(2 * PI * (scale * (x[i] - CENTER[i]))) for i in range(len(x))]
            )
        )
        + np.exp(1)
        + 20
    )


def eggholder(x: tuple) -> float:
    """
    This function computes the Eggholder test function with centering and rescaling.
    """
    scale = 1024  # See Ariguib et al. (2022 - Master thesis)
    return sum(
        [
            -scale
            * (x[i] - CENTER[i])
            * np.sin(
                np.sqrt(
                    np.abs(
                        scale * (x[i] - CENTER[i])
                        - scale * (x[i + 1] - CENTER[i + 1])
                        - 47
                    )
                )
            )
            - (scale * (x[i + 1] - CENTER[i + 1]) + 47)
            * np.sin(
                np.sqrt(
                    np.abs(
                        0.5 * scale * (x[i] - CENTER[i])
                        + scale * (x[i + 1] - CENTER[i + 1])
                        + 47
                    )
                )
            )
            for i in range(len(x) - 1)
        ]
    )


def sphere(x: tuple) -> float:
    """
    This function computes the 'sphere' test function with centering (no rescaling needed).
    """
    return sum([(x[i] - CENTER[i]) ** 2 for i in range(len(x))])


def rosenbrock(x: tuple) -> float:
    """
    This function computes the Rosenbrock test function with centering (no rescaling needed).
    """
    scale = 1
    return sum(
        [
            100
            * (scale * (x[i + 1] - CENTER[i + 1]) - (scale * (x[i] - CENTER[i])) ** 2)
            ** 2
            + (scale * (x[i] - CENTER[i]) - 1) ** 2
            for i in range(len(x) - 1)
        ]
    )


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
    ) - np.prod([scale * (x[i] - CENTER[i]) for i in range(len(x))])  # type: ignore


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### RBFs ###
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def thin_plate_spline(x: np.ndarray, center: np.ndarray) -> np.float32:
    r = np.linalg.norm(x - center)
    epsilon = np.finfo(np.float32).eps
    if r < epsilon:
        # Avoid singularity at 0
        return np.float32(0.0)
    else:
        return r**2 * np.log(r + epsilon)


def volume_spline(x: np.ndarray, center: np.ndarray) -> np.float32:
    r = np.linalg.norm(x - center)
    return np.float32(r)


def multiquadric(
    x: np.ndarray, center: np.ndarray, shape_parameter: np.float32
) -> np.float32:
    r = np.linalg.norm(x - center)
    return np.sqrt(r**2 + shape_parameter**2)


def inverse_multiquadric(
    x: np.ndarray, center: np.ndarray, shape_parameter: np.float32
) -> np.float32:
    return 1 / multiquadric(x, center, shape_parameter)


def gaussian(
    x: np.ndarray,
    center: np.ndarray,
    support_radius: float = np.inf,
    shape_parameter: float = 0.1,
) -> np.float32:
    r = np.linalg.norm(x - center)
    if not np.isinf(support_radius):
        # Compact support
        threshold = np.sqrt(-np.log(THRESHOLD) / shape_parameter)
        _support_radius = min(support_radius, threshold)
        _deltaY = np.exp(-((shape_parameter * _support_radius) ** 2))
        if r > _support_radius:
            return np.float32(0.0)
        else:
            return np.exp(-((shape_parameter * r) ** 2)) + _deltaY
    else:
        # Global support
        return np.exp(-((shape_parameter * r) ** 2))


def compact_thin_plate_splines(
    x: np.ndarray, center: np.ndarray, support_radius: float = 0.1
) -> np.float32:
    epsilon = np.finfo(np.float32).eps
    r = np.linalg.norm(x - center)
    r_norm = r / support_radius
    if r > support_radius:
        return np.float32(0.0)
    else:
        if r_norm < epsilon:
            return np.float32(1.0)
        else:
            return (
                1
                - 30 * r_norm**2
                - 10 * r_norm**3
                + 45 * r_norm**4
                - 6 * r_norm**5
                - 60 * r_norm**3 * np.log(r_norm + epsilon)
            )


def c0(x: np.ndarray, center: np.ndarray, support_radius: float) -> np.float32:
    r = np.linalg.norm(x - center)
    r_norm = r / support_radius
    if r > support_radius:
        return np.float32(0.0)
    else:
        return np.float32((1 - r_norm) ** 2)


def c2(x: np.ndarray, center: np.ndarray, support_radius: float) -> np.float32:
    r = np.linalg.norm(x - center)
    r_norm = r / support_radius
    if r > support_radius:
        return np.float32(0.0)
    else:
        return np.float32((1 - r_norm) ** 4 * (4 * r_norm + 1))


def c4(x: np.ndarray, center: np.ndarray, support_radius: float) -> np.float32:
    r = np.linalg.norm(x - center)
    r_norm = r / support_radius
    if r > support_radius:
        return np.float32(0.0)
    else:
        return np.float32((1 - r_norm) ** 6 * (35 * r_norm**2 + 18 * r_norm + 3))


def c6(x: np.ndarray, center: np.ndarray, support_radius: float) -> np.float32:
    r = np.linalg.norm(x - center)
    r_norm = r / support_radius
    if r > support_radius:
        return np.float32(0.0)
    else:
        return np.float32(
            (1 - r_norm) ** 8 * (32 * r_norm**3 + 25 * r_norm**2 + 8 * r_norm + 1)
        )


def c8(x: np.ndarray, center: np.ndarray, support_radius: float) -> np.float32:
    r = np.linalg.norm(x - center)
    r_norm = r / support_radius
    if r > support_radius:
        return np.float32(0.0)
    else:
        return np.float32(
            (1 - r_norm) ** 10
            * (
                1287 * r_norm**4
                + 1350 * r_norm**3
                + 630 * r_norm**2
                + 150 * r_norm
                + 15
            )
        )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### GRID GENERATION ###
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def generate_grid(dim: int = 2, resolution: int = 100) -> np.ndarray:
    _XMAX = 1.0
    _XMIN = 0.0
    _YMAX = 1.0
    _YMIN = 0.0
    if dim == 1:
        x = np.linspace(_XMIN, _XMAX, resolution)
        return x.reshape(-1, 1)
    elif dim == 2:
        x = np.linspace(_XMIN, _XMAX, resolution)
        y = np.linspace(_YMIN, _YMAX, resolution)
        return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    else:
        raise ValueError("Dimension must be 1 or 2.")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### OPTIMIZATION & RBF MAPPING ###
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def find_weights(
    function: Callable[..., np.float32],
    params: dict,
    test_function: Callable[..., float] | np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find the weights and polynomial coefficients for the RBF interpolant.
    """
    needs_shape_param = (
        True
        if function.__name__ in ["multiquadric", "inverse_multiquadric", "gaussian"]
        else False
    )
    needs_support_radius = (
        True
        if function.__name__
        in ["gaussian", "compact_thin_plate_splines", "c0", "c2", "c4", "c6", "c8"]
        else False
    )
    shape_parameter = params.get("shape_parameter", 0.1)
    support_radius = params.get(
        "support_radius", 0.1 if function.__name__ == "gaussian" else np.inf
    )  # Support for globally supported gaussian RBF
    C = np.zeros((len(grid), len(grid)))
    for i, xi in enumerate(grid):
        for j, xj in enumerate(grid):
            if needs_shape_param and needs_support_radius:
                C[i, j] = function(xi, xj, support_radius, shape_parameter)
            elif needs_shape_param and not needs_support_radius:
                C[i, j] = function(xi, xj, shape_parameter)
            elif needs_support_radius and not needs_shape_param:
                C[i, j] = function(xi, xj, support_radius)
            else:
                C[i, j] = function(xi, xj)

    # Create a P matrix with a column of ones and columns for each coordinate
    P = np.ones((len(grid), 1 + grid.shape[1]))
    P[:, 1:] = grid  # Add the coordinates

    if isinstance(test_function, np.ndarray):
        f = test_function
    else:
        f = np.array(
            [test_function(tuple(x)) for x in grid]
        )  # Evaluate the test function at each grid point

    # Determine the polynomial weights
    # beta = (beta_0, beta_l.T).T with beta_l being a vector of coefficients for the linear term (a component for each dimension), beta_0 is a constant term
    def objective(beta: np.ndarray) -> np.float32:
        return np.float32(np.linalg.norm(f - P @ beta))

    # Initial guess for beta weights
    beta_init = np.zeros(1 + grid.shape[1])

    # Minimize the objective function
    result = minimize(objective, beta_init, method="L-BFGS-B")
    if not result.success:
        raise ValueError("Optimization failed to find a solution.")

    beta = result.x

    # solve the linear system: C.lambda = f - P.beta for lambda
    lambda_ = np.linalg.solve(C, f - P @ beta)
    # Print condition number of the system
    condition_number = np.linalg.cond(C)
    # print(f"Condition number of the system: {condition_number:.2e}")

    return lambda_, beta, condition_number


def find_weights_for_opt(
    function: Callable[..., np.float32],
    shape_parameter: float,
    test_function: Callable[..., float] | np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find the weights and polynomial coefficients for the RBF interpolant.
    """
    # pairwise = cdist(grid, grid)
    C = np.zeros((len(grid), len(grid)))
    for i, xi in enumerate(grid):
        for j, xj in enumerate(grid):
            # Use the shape parameter directly in the RBF function
            if function.__name__ == "gaussian":
                C[i, j] = function(
                    xi, xj, support_radius=np.inf, shape_parameter=shape_parameter
                )
            else:
                C[i, j] = function(xi, xj, shape_parameter)
    # Create a P matrix with a column of ones and columns for each coordinate
    P = np.ones((len(grid), 1 + grid.shape[1]))
    P[:, 1:] = grid  # Add the coordinates
    if isinstance(test_function, np.ndarray):
        f = test_function
    else:
        f = np.array([test_function(tuple(x)) for x in grid])

    # Evaluate the test function at each grid point
    # Determine the polynomial weights
    def objective(beta: np.ndarray) -> np.float32:
        return np.float32(np.linalg.norm(f - P @ beta))

    # Initial guess for beta weights
    beta_init = np.zeros(1 + grid.shape[1])
    # Minimize the objective function
    result = minimize(objective, beta_init, method="L-BFGS-B")
    if not result.success:
        raise ValueError("Optimization failed to find a solution.")
    beta = result.x
    # solve the linear system: C.lambda = f - P.beta for lambda
    lambda_ = np.linalg.solve(C, f - P @ beta)
    condition_number = np.linalg.cond(C)
    return lambda_, beta, condition_number


def evaluate_rbf(
    function: Callable[..., np.float32],
    params: dict,
    grid: np.ndarray,
    weights: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the RBF function at the given grid points using the provided weights and beta coefficients.
    """
    centers = params.get("center", CENTER)
    shape_parameter = params.get("shape_parameter", 0.1)
    support_radius = params.get("support_radius", 0.1)
    needs_shape_param = function.__name__ in [
        "multiquadric",
        "inverse_multiquadric",
        "gaussian",
    ]
    needs_support_radius = function.__name__ in [
        "gaussian",
        "compact_thin_plate_splines",
        "c0",
        "c2",
        "c4",
        "c6",
        "c8",
    ]

    # Initialize result array with polynomial term
    result = np.zeros(len(grid))

    # Vectorized polynomial term calculation
    result = beta[0] + grid @ beta[1:]

    # Create a function that applies the right arguments
    def apply_rbf(point, center):
        if needs_shape_param and needs_support_radius:
            return function(point, center, support_radius, shape_parameter)
        elif needs_shape_param:
            return function(point, center, shape_parameter)
        elif needs_support_radius:
            return function(point, center, support_radius)
        else:
            return function(point, center)

    # Add weighted RBF terms using vectorized operations
    for i, point in enumerate(grid):
        rbf_values = np.array([apply_rbf(point, center) for center in centers])
        result[i] += np.sum(weights * rbf_values)

    return result


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### SHAPE PARAMETER OPTIMIZATION ###
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def optimize_shape_parameter(
    function: Callable[..., np.float32],
    test_function: Callable[..., float] | np.ndarray,
    grid: np.ndarray,
    start: float = 1e-5,
    end: float = 1.0,
    use_shgo: bool = True,
) -> tuple[float, np.ndarray]:
    if isinstance(test_function, np.ndarray):
        f = test_function
    else:
        f = np.array([test_function(tuple(x)) for x in grid])

    pairwise = cdist(grid, grid)

    def cost_function(shape_param: float):
        # Removed print for performance
        fname = function.__name__
        shape2 = shape_param**2  # Precompute for reuse

        # Use efficient NumPy operations
        if fname == "gaussian":
            A = np.exp(-np.square(shape_param * pairwise))
        elif fname == "multiquadric":
            A = np.sqrt(pairwise**2 + shape2)
        elif fname == "inverse_multiquadric":
            A = 1.0 / np.sqrt(pairwise**2 + shape2)
        else:
            raise ValueError("Function not supported for shape parameter optimization.")

        invA = np.linalg.pinv(A)
        Akk = np.einsum("ii->i", invA)  # Efficient diagonal extraction
        # Akk = np.diag(invA)
        # Use np.divide for safe element-wise division
        error_vector = np.divide(invA @ f, Akk, out=np.zeros_like(f), where=Akk != 0)
        return np.sum(np.abs(error_vector))

    if use_shgo:
        results = []
        res = shgo(
            cost_function,
            bounds=[(start, end)],
            n=100,  # Number of sampling points
            iters=5,  # Number of iterations
            minimizer_kwargs={"method": "L-BFGS-B"},
        )
        if not res.success:
            raise ValueError("Optimization failed to find a solution.")
        results = np.array([[res.x[0], res.fun]])
        optimal_shape_param = res.x[0]
    else:
        results = []
        search_space = np.linspace(start, end, 100)
        for shape_param in search_space:
            results.append([shape_param, cost_function(shape_param)])
        results = np.array(results)
        optimal_shape_param = results[np.argmin(results[:, 1]), 0]
    return optimal_shape_param, results


def empirical_shape_parameter(
    function: Callable[..., np.float32],
    test_function: Callable[..., float] | np.ndarray,
    grid: np.ndarray,
    grid_test: np.ndarray,
    start: float = 1e-5,
    end: float = 1.0,
) -> tuple[float, np.ndarray]:
    """
    Find the empirical optimal shape parameter for the RBF function.
    """
    if isinstance(test_function, np.ndarray):
        f = test_function
    else:
        f = np.array([test_function(tuple(x)) for x in grid])

    def cost_function(shape_param: float):
        lambda_, beta, _ = find_weights(
            function,
            {"shape_parameter": shape_param, "center": grid, "support_radius": np.inf},
            f,
            grid,
        )
        rbf_values = evaluate_rbf(
            function,
            {"shape_parameter": shape_param, "center": grid, "support_radius": np.inf},
            grid_test,
            lambda_,
            beta,
        )
        if isinstance(test_function, np.ndarray):
            return np.sqrt(np.mean((rbf_values - test_function) ** 2))
        else:
            return np.sqrt(
                np.mean(
                    (
                        rbf_values
                        - np.array([test_function(tuple(x)) for x in grid_test])
                    )
                    ** 2
                )
            )

    search_space = np.linspace(start, end, 100)
    results = []
    for shape_param in search_space:
        results.append([shape_param, cost_function(shape_param)])
    results = np.array(results)
    optimal_shape_param = results[np.argmin(results[:, 1]), 0]

    return optimal_shape_param, results


# def optimize_shape_parameter_with_polynomial_term(
#     function: Callable[..., np.float32],
#     test_function: Callable[..., float] | np.ndarray,
#     grid: np.ndarray,
#     start: float = 1e-5,
#     end: float = 1.0,
#     use_shgo: bool = True,
# ) -> tuple[float, np.ndarray]:
#     """
#     Optimize the shape parameter for the RBF function with polynomial term.
#     """
#     if isinstance(test_function, np.ndarray):
#         f = test_function
#     else:
#         f = np.array([test_function(tuple(x)) for x in grid])

#     pairwise = cdist(grid, grid)

#     def cost_function(shape_param: float):
#         if function.__name__ == "gaussian":
#             C = np.exp(-((shape_param * pairwise) ** 2))
#         elif function.__name__ == "multiquadric":
#             C = np.sqrt(pairwise**2 + shape_param**2)
#         elif function.__name__ == "inverse_multiquadric":
#             C = 1.0 / np.sqrt(pairwise**2 + shape_param**2)
#         else:
#             raise ValueError("Function not supported for shape parameter optimization.")
#         P = np.ones((len(grid), 1 + grid.shape[1]))
#         P[:, 1:] = grid
#         error_vector = np.zeros(len(grid))
#         for k, xk in enumerate(grid):
#             # Remove kth row and column from C
#             C_k = np.delete(np.delete(C, k, axis=0), k, axis=1)
#             f_k = np.delete(f, k)
#             P_k = np.delete(P, k, axis=0)
#             # Solve the linear system for lambda
#             beta = np.linalg.lstsq(P_k, f_k, rcond=None)[0]
#             lambda_ = np.linalg.solve(C_k, f_k - P_k @ beta)
#             # Evaluate the RBF at xk
#             # grid without xk
#             centers = np.delete(grid, k, axis=0)
#             for i, xi in enumerate(centers):
#                 if function.__name__ == "gaussian":
#                     rbf_value = function(
#                         xi, xk, support_radius=np.inf, shape_parameter=shape_param
#                     )
#                 else:
#                     rbf_value = function(xi, xk, shape_parameter=shape_param)
#                 error_vector[k] += lambda_[i] * rbf_value
#             error_vector[k] += beta[0] + xk @ beta[1:]
#         return np.sum(np.abs((error_vector - f)))

#     if use_shgo:
#         results = []
#         res = shgo(
#             cost_function,
#             bounds=[(start, end)],
#             n=100,  # Number of sampling points
#             iters=5,  # Number of iterations
#             minimizer_kwargs={"method": "L-BFGS-B"},
#         )
#         if not res.success:
#             raise ValueError("Optimization failed to find a solution.")
#         results = np.array([[res.x[0], res.fun]])
#         optimal_shape_param = res.x[0]
#     else:
#         results = []
#         search_space = np.linspace(start, end, 100)
#         for shape_param in search_space:
#             results.append([shape_param, cost_function(shape_param)])
#         results = np.array(results)
#         optimal_shape_param = results[np.argmin(results[:, 1]), 0]
#     return optimal_shape_param, results


def optimize_shape_parameter_with_polynomial_term(
    function: Callable[..., np.float32],
    test_function: Callable[..., float] | np.ndarray,
    grid: np.ndarray,
    start: float = 1e-5,
    end: float = 1.0,
    use_shgo: bool = True,
) -> tuple[float, np.ndarray]:
    if isinstance(test_function, np.ndarray):
        f = test_function
    else:
        f = np.array([test_function(tuple(x)) for x in grid])
    pairwise = cdist(grid, grid)
    P = np.ones((len(grid), 1 + grid.shape[1]))
    P[:, 1:] = grid  # Add the coordinates
    beta_init = np.zeros(1 + grid.shape[1])

    # Determine the polynomial weights
    def objective(beta: np.ndarray) -> np.float32:
        return np.float32(np.linalg.norm(P @ beta - f))

    # Minimize the objective function
    result = minimize(objective, beta_init, method="L-BFGS-B")
    if not result.success:
        raise ValueError("Optimization failed to find a solution.")
    beta = result.x
    polynomial_term = P @ beta

    def cost_function(shape_param: float):
        if function.__name__ == "gaussian":
            C = np.exp(-((shape_param * pairwise) ** 2))
        elif function.__name__ == "multiquadric":
            C = np.sqrt(pairwise**2 + shape_param**2)
        elif function.__name__ == "inverse_multiquadric":
            C = 1.0 / np.sqrt(pairwise**2 + shape_param**2)
        else:
            raise ValueError("Function not supported for shape parameter optimization.")
        error_vector = np.zeros(len(grid))
        for k, xk in enumerate(grid):
            # Remove kth row and column from C
            C_k = np.delete(np.delete(C, k, axis=0), k, axis=1)
            f_k = np.delete(f, k)
            P_k = np.delete(P, k, axis=0)
            # Solve the linear system for lambda
            lambda_ = np.linalg.solve(C_k, f_k - P_k @ beta)
            # Evaluate the RBF at xk
            centers = np.delete(grid, k, axis=0)
            for i, xi in enumerate(centers):
                if function.__name__ == "gaussian":
                    rbf_value = function(
                        xi, xk, support_radius=np.inf, shape_parameter=shape_param
                    )
                else:
                    rbf_value = function(xi, xk, shape_parameter=shape_param)
                error_vector[k] += lambda_[i] * rbf_value
            error_vector[k] += polynomial_term[k]
        return np.sum(np.abs((error_vector - f)))

    if use_shgo:
        results = []
        res = shgo(
            cost_function,
            bounds=[(start, end)],
            n=100,  # Number of sampling points
            iters=5,  # Number of iterations
            minimizer_kwargs={"method": "L-BFGS-B"},
        )
        if not res.success:
            raise ValueError("Optimization failed to find a solution.")
        results = np.array([[res.x[0], res.fun]])
        optimal_shape_param = res.x[0]
    else:
        results = []
        search_space = np.linspace(start, end, 100)
        for shape_param in search_space:
            results.append([shape_param, cost_function(shape_param)])
        results = np.array(results)
        optimal_shape_param = results[np.argmin(results[:, 1]), 0]
    return optimal_shape_param, results


if __name__ == "__main__":
    grid = generate_grid(dim=1, resolution=128)
    target_grid = generate_grid(dim=1, resolution=64)
    function = multiquadric
    test_function = ackley

    shape_parameters = []
    opt, results = optimize_shape_parameter(
        function,
        test_function,
        grid,
        start=1e-5,
        end=1.0,
        use_shgo=True,
    )
    print(f"Optimal shape parameter: {opt:.4f}")
    shape_parameters.append(opt)

    opt, results = empirical_shape_parameter(
        function,
        test_function,
        grid,
        target_grid,
        start=1e-5,
        end=1.0,
    )
    print(f"Empirical optimal shape parameter: {opt}")
    shape_parameters.append(opt)

    # opt, results = optimize_shape_parameter_with_polynomial_term(
    #     function,
    #     test_function,
    #     grid,
    #     start=1e-5,
    #     end=1.0,
    #     use_shgo=True,
    # )
    # print(f"Optimal shape parameter with polynomial term: {opt}")
    # shape_parameters.append(opt)

    # for shape_param in shape_parameters:
    #     lambda_, beta, _ = find_weights(
    #         function,
    #         {"shape_parameter": shape_param, "center": grid, "support_radius": np.inf},
    #         test_function,
    #         grid,
    #     )
    #     rbf_values = evaluate_rbf(
    #         function,
    #         {"shape_parameter": shape_param, "center": grid, "support_radius": np.inf},
    #         target_grid,
    #         lambda_,
    #         beta,
    #     )
    #     error = np.sqrt(
    #         np.mean(
    #             (rbf_values - np.array([test_function(tuple(x)) for x in target_grid]))
    #             ** 2
    #         )
    #     )
    #     print(f"Error for shape parameter {shape_param}: {error}")
    # # plt.plot(results[:, 0], results[:, 1])
    # # plt.xlabel("Shape Parameter")
    # # plt.ylabel("Cost Function Value")
    # # plt.show()
