## Standard librairies
import pathlib
import subprocess
import json

## Third-party librairies
import vtk

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.interpolate import griddata

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Interpolation test functions
# * With rescale + centering on CENTER
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def sphere(x: tuple) -> float:
    return sum([(x[i] - CENTER[i]) ** 2 for i in range(len(x))])


def drop_wave(x: tuple) -> float:
    scale = 1.5
    return 1 - (
        1
        + np.cos(
            12
            * np.sqrt(sum([(scale * (x[i] - CENTER[i])) ** 2 for i in range(len(x))]))
        )
    ) / (0.5 * sum([(scale * (x[i] - CENTER[i])) ** 2 for i in range(len(x))]) + 2)


def ackley(x: tuple) -> float:
    scale = 5
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


def rosenbrock(x: tuple) -> float:
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


def rastigrin(x: tuple) -> float:
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
    ) - (scale * (x[0] - CENTER[0])) * (scale * (x[2] - CENTER[2]))


def eggholder(x: tuple) -> float:
    scale = 1024
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Constants, paths and file names
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PATH_TO_MESHES = pathlib.Path("./meshes")
PATH_TO_OUT = pathlib.Path("./out")
# STRUCTURE_MESH_NAME = "fluid_nodes_fastest_64.vtu"
# FLUID_NODES_MESH_NAME = "fluid_nodes_fastest_128.vtu"
# FLUID_CENTERS_MESH_NAME = "fluid_centers_fastest_64.vtu"
STRUCTURE_MESH_NAME = "0_01.vtk"
FLUID_NODES_MESH_NAME = "0_02.vtk"
FLUID_CENTERS_MESH_NAME = "0_02.vtk"
GLOBAL_INPUT_MESH = FLUID_NODES_MESH_NAME
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Simulation parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FUNCTION = rastigrin
MAPPING = "NN"
BLADE = True if "vtk" in STRUCTURE_MESH_NAME else False
CENTER = (0.5, 0.0, 0.5)
ENABLE_GRADIENT = True if function.__name__ in ["rosenbrock", "sphere", "rastigrin"] else False # Enable gradient option in precice-aste-evalute (rastigrin, sphere, rosenbrock only)
MODIFIER = (
    "fc_to_sn"
    + f"_s{STRUCTURE_MESH_NAME.split('.')[0].split('_')[-1]}_{FUNCTION.__name__}_{MAPPING}_B{BLADE}"
)
p = 1 / 2.54 # Figsize parameter for matplotlib

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Parameters for ASTE
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEFAULT_NB_PROCS = 2
DEFAULT_DATA_NAME = "InputData"
DEFAULT_INTERPOLATED_DATA_NAME = "InterpolatedData"


def read_vtu(path_to_vtu: pathlib.Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path_to_vtu))
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()
    point_coords = np.array(
        [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    )
    return point_coords


def find_min_max(func, grid: np.array):
    values = np.array([func(tuple(point)) for point in grid])
    minimum = np.min(values)
    maximum = np.max(values)
    return minimum, maximum


def linear_scaling(
    func, x: tuple, m: float, M: float, to_string_mode: bool = False, f: str = None
):
    offset = 1
    if to_string_mode and f != None:
        return f"({f}-{m})/({M}-{m})+{offset}"
    else:
        return (func(tuple(x)) - m) / (M - m) + offset


def generate_grid(n: int, xmax: float = 1.0, ymax: float = 1.0) -> np.array:
    x = np.linspace(0, xmax, n)
    z = np.linspace(0, ymax, n)
    X, Z = np.meshgrid(x, z, indexing="xy")
    Y = np.zeros_like(X)
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def evaluate(
    f,
    grid: np.array,
    do_scaling: bool = True,
    minimum: float = None,
    maximum: float = None,
) -> np.array:
    if do_scaling:
        return np.array(
            [linear_scaling(f, tuple(point), minimum, maximum) for point in grid]
        )
    else:
        return np.array([f(tuple(point)) for point in grid])


def plot_contour_and_surface(
    filename: str, values: np.array, grid: np.array, levels: int = 15
):
    x = grid[:, 0]
    y = grid[:, 2]
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    X, Y = np.meshgrid(x_unique, y_unique, indexing="xy")
    Z = values.reshape((ny, nx))

    fig = plt.figure(figsize=(40 * p, 20 * p))

    ax1 = fig.add_subplot(1, 2, 1)
    contourf = ax1.contourf(X, Y, Z, levels=50, cmap="gnuplot")
    contour = ax1.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.7)
    ax1.clabel(contour, inline=True, fontsize=6, fmt="%.2f")
    fig.colorbar(contourf, ax=ax1, label="Function value")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    plt.gca().set_aspect("equal")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(X, Y, Z, cmap=cm.gnuplot, edgecolor="black", linewidth=0.3)
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, label="Function value")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_zlabel("$f(x, y)$")

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300)
    plt.close()


def plot(n: int, func, do_scaling: bool = True):
    grid = generate_grid(n)
    m, M = find_min_max(func, grid)
    values = evaluate(func, grid, do_scaling=True, minimum=m, maximum=M)
    plot_contour_and_surface(func.__name__, values, grid, levels=5)


def compute_integral(x: np.array, y: np.array, values: np.array):
    xi, yi = np.meshgrid(np.unique(x), np.unique(y))
    points = np.column_stack((x, y))
    grid_values = griddata(points, values, (xi, yi), method="linear")
    return np.trapz(np.trapz(grid_values, xi[0]), yi[:, 0])


if BLADE == "YES":
    DIMENSIONS = (1, 0, 1)
else:
    points = read_vtu(str(PATH_TO_MESHES / GLOBAL_INPUT_MESH))
    DIMENSIONS = (np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2]))
# SIZE = points[:, 0].shape[0]
SIZE = 100

# plot(100, rastigrin, do_scaling=True)


class Run:
    # Class constants
    DEFAULT_INPUT_MESH_NAME = "input_mesh.vtu"
    DEFAULT_MESH_NAME_A = "a_mesh"
    DEFAULT_MESH_NAME_B = "b_mesh"
    DEFAULT_OUTPUT_NAME = (
        "result_"
        + f"f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}"
        + f"_s{STRUCTURE_MESH_NAME.split('.')[0].split('_')[-1]}"
        + ".vtu"
    )

    def __init__(self, nb_process: int, evaluation_function: str):
        self.nb_process = nb_process
        if evaluation_function in ["franke3d", "eggholder3d", "rosenbrock3d"]:
            self.evaluation_function = evaluation_function
        elif evaluation_function == "sphere":
            minimum, maximum = find_min_max(
                sphere, generate_grid(SIZE, DIMENSIONS[0], DIMENSIONS[2])
            )
            self.evaluation_function = linear_scaling(
                sphere,
                None,
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(x-{CENTER[0]})^2+(y-{CENTER[1]})^2+(z-{CENTER[2]})^2",
            )
        elif evaluation_function == "drop_wave":
            SCALE = 1.5
            minimum, maximum = find_min_max(
                drop_wave, generate_grid(SIZE, DIMENSIONS[0], DIMENSIONS[2])
            )
            self.evaluation_function = linear_scaling(
                drop_wave,
                None,
                minimum,
                maximum,
                to_string_mode=True,
                f=f"1-((1+cos(12*sqrt(({SCALE}*(x-{CENTER[0]}))^2+({SCALE}*(y-{CENTER[1]}))^2+({SCALE}*(z-{CENTER[2]}))^2)))/(0.5*(({SCALE}*(x-{CENTER[0]}))^2+({SCALE}*(y-{CENTER[1]}))^2+({SCALE}*(z-{CENTER[2]}))^2)+2))",
            )
        elif evaluation_function == "ackley":
            SCALE = 5
            DIM = 3
            PI = 3.141259
            minimum, maximum = find_min_max(
                ackley, generate_grid(SIZE, DIMENSIONS[0], DIMENSIONS[2])
            )
            self.evaluation_function = linear_scaling(
                ackley,
                None,
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(-20)*exp(-0.2*sqrt(1/{DIM}*(({SCALE}*(x-{CENTER[0]}))^2+({SCALE}*(y-{CENTER[1]}))^2+({SCALE}*(z-{CENTER[2]}))^2)))-exp(1/{DIM}*(cos(2*{PI}*{SCALE}*(x-{CENTER[0]}))+cos(2*{PI}*{SCALE}*(y-{CENTER[1]}))+cos(2*{PI}*{SCALE}*(z-{CENTER[2]}))))+exp(1)+20",
            )
        elif evaluation_function == "rosenbrock":
            SCALE = 1
            minimum, maximum = find_min_max(
                rosenbrock, generate_grid(SIZE, DIMENSIONS[0], DIMENSIONS[2])
            )
            self.evaluation_function = linear_scaling(
                rosenbrock,
                None,
                minimum,
                maximum,
                to_string_mode=True,
                f=f"100*(({SCALE}*(y-{CENTER[1]})-({SCALE}*(x-{CENTER[0]}))^2)^2+({SCALE}*(z-{CENTER[2]})-({SCALE}*(y-{CENTER[1]}))^2)^2)+({SCALE}*(x-{CENTER[0]})-1)^2+({SCALE}*(y-{CENTER[1]})-1)^2",
            )
        elif evaluation_function == "eggholder":
            SCALE = 1024
            minimum, maximum = find_min_max(
                eggholder, generate_grid(SIZE, DIMENSIONS[0], DIMENSIONS[2])
            )
            self.evaluation_function = linear_scaling(
                eggholder,
                None,
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(0-{SCALE}*(x-{CENTER[0]}))*sin(sqrt(abs({SCALE}*(x-{CENTER[0]})-{SCALE}*(y-{CENTER[1]})-47)))-({SCALE}*(y-{CENTER[1]})+47)*sin(sqrt(abs(0.5*{SCALE}*(x-{CENTER[0]})+{SCALE}*(y-{CENTER[1]})+47)))-{SCALE}*(y-{CENTER[1]})*sin(sqrt(abs({SCALE}*(y-{CENTER[1]})-{SCALE}*(z-{CENTER[2]})-47)))-({SCALE}*(z-{CENTER[2]})+47)*sin(sqrt(abs(0.5*{SCALE}*(y-{CENTER[1]})+{SCALE}*(z-{CENTER[2]})+47)))",
            )
        elif evaluation_function == "rastigrin":
            SCALE = 10
            minimum, maximum = find_min_max(
                rastigrin, generate_grid(SIZE, DIMENSIONS[0], DIMENSIONS[2])
            )
            PI = 3.14159
            self.evaluation_function = linear_scaling(
                rastigrin,
                None,
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(({SCALE}*(x-{CENTER[0]}))^2-10*cos(2*{PI}*{SCALE}*(x-{CENTER[0]})))+(({SCALE}*(y-{CENTER[1]}))^2-10*cos(2*{PI}*{SCALE}*(y-{CENTER[1]})))+(({SCALE}*(z-{CENTER[2]}))^2-10*cos(2*{PI}*{SCALE}*(z-{CENTER[2]})))-({SCALE}*(x-{CENTER[0]}))*({SCALE}*(z-{CENTER[2]}))",
            )
        else:
            raise ValueError("Unknown function.")

    def __str__(self) -> str:
        return f"Run(nb = {self.nb_process}, func =  {self.evaluation_function})"

    def _run_command(self, cmd: str, dry_run: bool = False):
        if dry_run:
            print(cmd)
        else:
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Command failed: {cmd}")
                print(e)

    def _run_command_parallel(self, cmd1: str, cmd2: str, dry_run: bool = False):
        if dry_run:
            print(cmd1)
            print(cmd2)
        else:
            try:
                p1 = subprocess.Popen(cmd1, shell=True)
                p2 = subprocess.Popen(cmd2, shell=True)
                p1.wait()
                p2.wait()
            except Exception as e:
                print(f"[ERROR] Parallel command execution failed:\n{cmd1}\n{cmd2}")
                print(e)

    def evaluate(self, path_to_mesh: pathlib.Path) -> None:
        if ENABLE_GRADIENT:
            cmd = f'precice-aste-evaluate -m {path_to_mesh} -f "{self.evaluation_function}" -d "{DEFAULT_DATA_NAME}" -o {self.DEFAULT_INPUT_MESH_NAME} --gradient --log DEBUG'
        else:
            cmd = f'precice-aste-evaluate -m {path_to_mesh} -f "{self.evaluation_function}" -d "{DEFAULT_DATA_NAME}" -o {self.DEFAULT_INPUT_MESH_NAME} --gradient --log DEBUG'
        self._run_command(cmd)

    def partition(
        self,
        path_to_mesh: pathlib.Path,
        mesh_a_name: str = DEFAULT_MESH_NAME_A,
        mesh_b_name: str = DEFAULT_MESH_NAME_B,
    ) -> None:
        cmd_a = f"precice-aste-partition -m {self.DEFAULT_INPUT_MESH_NAME} -n {self.nb_process} -o {mesh_a_name} --dir {mesh_a_name} --algorithm uniform"
        cmd_b = f"precice-aste-partition -m {path_to_mesh} -n {self.nb_process} -o {mesh_b_name} --dir {mesh_b_name} --algorithm uniform"
        self._run_command_parallel(cmd_a, cmd_b)

    def run(self):
        path_to_mapped = pathlib.Path("mapped/")
        path_to_mapped.mkdir(parents=True, exist_ok=True)
        cmd_a = f'mpirun -n {self.nb_process} precice-aste-run -p A --mesh {self.DEFAULT_MESH_NAME_A}/{self.DEFAULT_MESH_NAME_A} --data "{DEFAULT_DATA_NAME}"'
        cmd_b = f'mpirun -n {self.nb_process} precice-aste-run -p B --mesh {self.DEFAULT_MESH_NAME_B}/{self.DEFAULT_MESH_NAME_B} --output mapped/mapped --data "{DEFAULT_INTERPOLATED_DATA_NAME}"'
        self._run_command_parallel(cmd_a, cmd_b)

    def join(self):
        cmd = f"precice-aste-join -m mapped/mapped -o {self.DEFAULT_OUTPUT_NAME} --recovery {self.DEFAULT_MESH_NAME_B}/{self.DEFAULT_MESH_NAME_B}_recovery.json"
        self._run_command(cmd)

    def stats(self):
        cmd = f'precice-aste-evaluate -m {self.DEFAULT_OUTPUT_NAME} -f "{self.evaluation_function}" -d "Error" --diffdata "{DEFAULT_INTERPOLATED_DATA_NAME}" --diff --stats --log DEBUG'
        self._run_command(cmd)

    def _vtu_to_csv(self, path_to_file: str, filename: str, data: str):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path_to_file)
        reader.Update()
        mesh = reader.GetOutput()
        points = mesh.GetPoints()
        point_coords = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        )
        data_array = mesh.GetPointData().GetArray(data)
        if data_array is None:
            raise ValueError(f"Could not find {data} in the VTU file.")
        # Get the information
        data_values = np.array(
            [data_array.GetValue(i) for i in range(data_array.GetNumberOfTuples())]
        )
        x = point_coords[:, 0]
        y = point_coords[:, 1]
        z = point_coords[:, 2]
        # Write to file
        PATH_TO_OUT.mkdir(parents=True, exist_ok=True)
        output_array = np.column_stack((x, y, z, data_values))
        if MODIFIER != None:
            path_to_file = str(
                PATH_TO_OUT
                / f"{filename}_f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}_{MODIFIER}.csv"
            )
        else:
            path_to_file = str(
                PATH_TO_OUT
                / f"{filename}_f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}.csv"
            )
        np.savetxt(
            path_to_file,
            output_array,
            delimiter=",",
            header=f"x,y,z,{data}",
            comments="",
        )

    def save_results(self):
        # Read and process results
        self._vtu_to_csv(self.DEFAULT_INPUT_MESH_NAME, "input", DEFAULT_DATA_NAME)
        self._vtu_to_csv(
            self.DEFAULT_OUTPUT_NAME, "output", DEFAULT_INTERPOLATED_DATA_NAME
        )

    def clean(self):
        self._run_command("make clean")


class Process:
    DEFAULT_INPUT_CSV_NAME = str(
        PATH_TO_OUT
        / f"input_f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}_{MODIFIER}.csv"
    )
    DEFAULT_OUTPUT_CSV_NAME = str(
        PATH_TO_OUT
        / f"output_f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}_{MODIFIER}.csv"
    )
    DEFAULT_STATS_JSON_FILE = (
        f"result_f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}.json"
    )

    def __init__(
        self,
        input_csv_file: pathlib.Path = DEFAULT_INPUT_CSV_NAME,
        output_csv_file: pathlib.Path = DEFAULT_OUTPUT_CSV_NAME,
        stats_file: pathlib.Path = DEFAULT_STATS_JSON_FILE,
    ) -> None:
        self.input_csv_file = input_csv_file
        self.output_csv_file = output_csv_file
        self.stats_file = stats_file

    def read_csv(self, which: str = "out"):
        # Expected format: x, y, z, data_values
        if which == "in":
            x, y, z, values = np.loadtxt(
                self.input_csv_file, unpack=True, skiprows=1, delimiter=","
            )
        else:
            x, y, z, values = np.loadtxt(
                self.output_csv_file, unpack=True, skiprows=1, delimiter=","
            )
        return x, y, z, values

    def compute_error_metrics(self, func):
        x_in, y_in, z_in, values_in = self.read_csv(which="in")
        x_out, y_out, z_out, values_out = self.read_csv(which="out")
        n = x_out.size
        minimum, maximum = find_min_max(
            func, generate_grid(SIZE, DIMENSIONS[0], DIMENSIONS[2])
        )
        predicted = np.array(
            [
                linear_scaling(func, (x, y, z), minimum, maximum)
                for x, y, z in zip(x_out, y_out, z_out)
            ]
        )
        relative_errors_pointwise = 100 * np.abs((predicted - values_out) / predicted)
        linfty_global = np.max(np.abs(values_out - predicted))
        rmse = np.sqrt(1 / n * np.sum((values_out - predicted) ** 2))
        integral_in = compute_integral(x_in, z_in, values_in)
        integral_out = compute_integral(x_out, z_out, values_out)
        return (
            x_out,
            y_out,
            z_out,
            values_out,
            relative_errors_pointwise,
            linfty_global,
            rmse,
            integral_in,
            integral_out,
        )

    def read_stats(self) -> "dict[str, float]":
        with open(self.stats_file, "r") as f:
            data = json.load(f)
        return data

    def plot_data(self, func):
        # Load data
        (
            x_out,
            _,
            z_out,
            output_data,
            relative_errors,
            linfty,
            rmse,
            integral_in,
            integral_out,
        ) = self.compute_error_metrics(func)
        print(f"L_infty={linfty}, RMSE(global)={rmse}")
        print(integral_in, integral_out)

        # Get unique x and y values
        x_unique = np.unique(x_out)
        y_unique = np.unique(z_out)

        # Sort and reshape data to 2D grid
        sorted_indices = np.lexsort((x_out, z_out))
        output_2d = output_data[sorted_indices].reshape(len(y_unique), len(x_unique))
        relative_errors_2d = relative_errors[sorted_indices].reshape(
            len(y_unique), len(x_unique)
        )

        # Set up the 2D plot
        fontsize = 20
        labelsize = 15
        fig, axes = plt.subplots(1, 2, figsize=(50 * p, 22 * p))

        # Plot the left half (output_2d)
        left_plot = axes[0].imshow(
            output_2d,
            extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
            origin="lower",
            aspect="auto",
            cmap="gnuplot",
        )
        axes[0].tick_params(labelsize=labelsize)
        cbar_left = fig.colorbar(left_plot, ax=axes[0], label="Mapped Values")
        cbar_left.ax.tick_params(labelsize=labelsize)
        cbar_left.set_label("Mapped Values", fontsize=fontsize)  # Change font size here
        axes[0].set_aspect("equal")
        axes[0].set_title("Mapped Values", fontsize=fontsize)
        axes[0].set_xlabel("$x$", fontsize=fontsize)
        axes[0].set_ylabel("$y$", fontsize=fontsize)

        # Plot the right half (relative_errors_2d)
        right_plot = axes[1].imshow(
            relative_errors_2d,
            extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]],
            origin="lower",
            aspect="auto",
            cmap="binary",
        )
        axes[1].tick_params(labelsize=labelsize)
        cbar_right = fig.colorbar(right_plot, ax=axes[1], label="Relative Errors")
        cbar_right.set_label(
            "Relative Errors", fontsize=fontsize
        )  # Change font size here
        cbar_right.ax.tick_params(labelsize=labelsize)
        axes[1].set_aspect("equal")
        axes[1].set_title(
            "Relative Error $\\varepsilon = 100\\times\\left\\vert\\frac{v_f - v_m}{v_f}\\right\\vert$",
            fontsize=fontsize,
        )
        axes[1].set_xlabel("$x$", fontsize=fontsize)
        # decimals = 3
        # plt.suptitle(
        #     f"$L_\infty$={linfty:.{decimals}e}, RMSE(global)={rmse:.{decimals}e}",
        #     fontsize=fontsize,
        # )

        # Adjust layout for better appearance
        plt.tight_layout()
        plt.savefig(
            f"./images/errors_f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}_{MODIFIER}.pdf"
        )
        plt.close()


if __name__ == "__main__":
    test = Run(2, FUNCTION.__name__)
    test.evaluate(PATH_TO_MESHES / STRUCTURE_MESH_NAME)
    test.partition(PATH_TO_MESHES / FLUID_NODES_MESH_NAME)
    test.run()
    test.join()
    test.stats()
    test.save_results()
    # test_plot = Process()
    # test_plot.plot_data(FUNCTION)
    # print(find_extrema(eggholder))
    # compares_nodes(pathlib.Path("./meshes/fluid_nodes_fastest_64.vtu"), pathlib.Path("./checks/FSI_interface_mesh_for_preCICE_A-patch_nodes_proc_0001.dat"))
    # print(np.max(plot_on_mesh(pathlib.Path("./meshes/structure_nodes_calculix_32.vtu"),drop_wave)))
