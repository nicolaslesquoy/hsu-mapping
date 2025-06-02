## Standard librairies
import pathlib
import subprocess
import shutil
import json
from typing import Optional

## Third-party librairies
# * pip install -r requirements.txt
import vtk
import numpy as np
import matplotlib.pyplot as plt

p = 1 / 2.54

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## FUNCTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ? Center of the computational domain for lid-driven cavity
CENTER = (0.5, 0.0, 0.5)


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
    ) - (scale * (x[0] - CENTER[0])) * (scale * (x[2] - CENTER[2]))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## CONSTANTS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PATH_TO_MESHES = pathlib.Path("./meshes")
assert PATH_TO_MESHES.exists(), "Mesh folder not found."
if not PATH_TO_MESHES.exists():
    PATH_TO_MESHES.mkdir(parents=True, exist_ok=True)
    print(f"Mesh folder created: {PATH_TO_MESHES}")
PATH_TO_STRETCHED = pathlib.Path("./stretched")
assert PATH_TO_STRETCHED.exists(), "Stretched folder not found."
if not PATH_TO_STRETCHED.exists():
    PATH_TO_STRETCHED.mkdir(parents=True, exist_ok=True)
    print(f"Stretched folder created: {PATH_TO_STRETCHED}")
PATH_TO_OUT = pathlib.Path("./out")
assert PATH_TO_OUT.exists(), "Output folder not found."
if not PATH_TO_OUT.exists():
    PATH_TO_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output folder created: {PATH_TO_OUT}")
PATH_TO_TEMPLATES = pathlib.Path("./templates")
DEFAULT_NB_PROCS = 4
DEFAULT_DATA_NAME = "InputData"
DEFAULT_INTERPOLATED_DATA_NAME = "InterpolatedData"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## ConfigParser + related functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_json(path_to_file: pathlib.Path) -> dict:
    """
    This function reads a given JSON file.
    """
    with open(path_to_file, "r") as file:
        data = json.load(file)
    return data


def is_blade(path_to_file: pathlib.Path) -> bool:
    """
    This function checks based on the file extension if a file represents a wind turbine blade or not.
    """
    return True if "vtk" in path_to_file.name else False


def parse_parameters(method: str, parameters: dict) -> tuple[str, float] | None:
    if "rbf" not in method:
        return None
    else:
        basis_function: str = parameters["basis-function"]
        support_radius: float = float(parameters["support-radius"])
    return (basis_function, support_radius)


class ConfigParser:
    """
    This class parses a JSON config file for a given case.
    """

    def __init__(self, path_to_config: pathlib.Path) -> None:
        assert path_to_config.exists(), "Configuration file not found."
        config = read_json(path_to_config)
        self.path_to_config: pathlib.Path = path_to_config
        self.input_mesh: pathlib.Path = PATH_TO_MESHES / config["input-mesh"]
        assert self.input_mesh.exists(), "Input mesh not found."
        self.output_mesh: pathlib.Path = PATH_TO_MESHES / config["output-mesh"]
        assert self.output_mesh.exists(), "Output mesh not found"
        self.is_blade: bool = is_blade(self.input_mesh)
        self.test_function: str = config["test-function"]
        assert self.test_function in [
            "sphere",
            "ackley",
            "rosenbrock",
            "eggholder",
            "rastrigin_mod",
            "drop_wave",
        ], "Unsupported function."
        self.mapping_method: str = config["mapping-method"]
        self.additional_parameters: tuple[str, float] | None = parse_parameters(
            self.mapping_method, config["additional-config"]
        )
        self.number_of_processes: int = config["nb-procs"]

    def __str__(self) -> str:
        return f"ConfigParser({self.path_to_config})"

    def print_to_terminal(self) -> None:
        with open(self.path_to_config, "r") as file:
            json_object = json.load(file)
        print(json.dumps(json_object, indent=2))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## XMLEditor + related functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_txt(path_to_file: pathlib.Path) -> list:
    with open(path_to_file, "r") as file:
        content = file.readlines()
    return content


class XMLEditor:
    """
    This class edits the precice-config.xml file based on a case config.
    """

    def __init__(self, config: "ConfigParser") -> None:
        self.mapping_method: str = config.mapping_method
        if config.additional_parameters is not None:
            self.additional_parameters: tuple[str, float] = config.additional_parameters
        if "rbf" in self.mapping_method:
            self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / "rbf.txt"
        else:
            self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / "no_rbf.txt"

    def edit(self) -> str:
        lines = read_txt(self.path_to_template)
        content: str = "".join(lines)
        if "rbf" in self.mapping_method:
            content = content.replace("$rbf_type$", self.mapping_method)
            assert "$rbf_type$" not in content
            content = content.replace("$basis-function$", self.additional_parameters[0])
            assert "$basis-function$" not in content
            content = content.replace("$radius$", str(self.additional_parameters[1]))
            assert "$radius$" not in content
        else:
            content = content.replace("$method$", self.mapping_method)
            assert "$method$" not in content
        return content

    def write(self) -> None:
        content = self.edit()
        with open("precice-config.xml", "w") as file:
            file.write(content)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Run + related functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def read_vtu(path_to_vtu: pathlib.Path):
    reader = vtk.vtkXMLUnstructuredGridReader()  # type: ignore
    reader.SetFileName(str(path_to_vtu))
    reader.Update()
    mesh = reader.GetOutput()
    points = mesh.GetPoints()
    point_coords = np.array(
        [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    )
    return point_coords


def find_min_max(func, grid: np.ndarray):
    values = np.array([func(tuple(point)) for point in grid])
    minimum = np.min(values)
    maximum = np.max(values)
    return minimum, maximum


def linear_scaling(
    func,
    x: tuple,
    m: float,
    M: float,
    to_string_mode: bool = False,
    f: Optional[str] = None,
):
    offset = 1
    if to_string_mode and f is not None:
        return f"({f}-{m})/({M}-{m})+{offset}"
    else:
        return (func(tuple(x)) - m) / (M - m) + offset


def generate_grid(n: int, xmax: float = 1.0, ymax: float = 1.0) -> np.ndarray:
    x = np.linspace(0, xmax, n)
    z = np.linspace(0, ymax, n)
    X, Z = np.meshgrid(x, z, indexing="xy")
    Y = np.zeros_like(X)
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def evaluate(
    f,
    grid: np.ndarray,
    do_scaling: bool = True,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> np.ndarray:
    if do_scaling and minimum is not None and maximum is not None:
        return np.array(
            [linear_scaling(f, tuple(point), minimum, maximum) for point in grid]
        )
    else:
        return np.array([f(tuple(point)) for point in grid])


class Run:
    # Class constants
    DEFAULT_INPUT_MESH_NAME = pathlib.Path("input_mesh.vtu")
    DEFAULT_OUTPUT_MESH_NAME = pathlib.Path("result.vtu")
    DEFAULT_MESH_NAME_A = "a_mesh"
    DEFAULT_MESH_NAME_B = "b_mesh"
    # DEFAULT_OUTPUT_MESH_NAME = (
    #     "result_"
    #     + f"f{FLUID_NODES_MESH_NAME.split('.')[0].split('_')[-1]}"
    #     + f"_s{STRUCTURE_MESH_NAME.split('.')[0].split('_')[-1]}"
    #     + ".vtu"
    # )
    SIZE = 100

    def __init__(
        self,
        config: "ConfigParser",
        enable_gradient: bool = False,
        output_name: Optional[str] = None,
    ):
        self.parameters: "ConfigParser" = config
        self.nb_process = config.number_of_processes
        if config.test_function in ["franke3d", "eggholder3d", "rosenbrock3d"]:
            self.evaluation_function = config.test_function
        elif config.test_function == "sphere":
            minimum, maximum = find_min_max(sphere, generate_grid(self.SIZE))
            self.evaluation_function = linear_scaling(
                sphere,
                (0, 0, 0),
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(x-{CENTER[0]})^2+(y-{CENTER[1]})^2+(z-{CENTER[2]})^2",
            )
        elif config.test_function == "drop_wave":
            SCALE = 1.5
            minimum, maximum = find_min_max(drop_wave, generate_grid(self.SIZE))
            self.evaluation_function = linear_scaling(
                drop_wave,
                (0, 0, 0),
                minimum,
                maximum,
                to_string_mode=True,
                f=f"1-((1+cos(12*sqrt(({SCALE}*(x-{CENTER[0]}))^2+({SCALE}*(y-{CENTER[1]}))^2+({SCALE}*(z-{CENTER[2]}))^2)))/(0.5*(({SCALE}*(x-{CENTER[0]}))^2+({SCALE}*(y-{CENTER[1]}))^2+({SCALE}*(z-{CENTER[2]}))^2)+2))",
            )
        elif config.test_function == "ackley":
            SCALE = 5
            DIM = 3
            PI = 3.141259
            minimum, maximum = find_min_max(ackley, generate_grid(self.SIZE))
            self.evaluation_function = linear_scaling(
                ackley,
                (0, 0, 0),
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(-20)*exp(-0.2*sqrt(1/{DIM}*(({SCALE}*(x-{CENTER[0]}))^2+({SCALE}*(y-{CENTER[1]}))^2+({SCALE}*(z-{CENTER[2]}))^2)))-exp(1/{DIM}*(cos(2*{PI}*{SCALE}*(x-{CENTER[0]}))+cos(2*{PI}*{SCALE}*(y-{CENTER[1]}))+cos(2*{PI}*{SCALE}*(z-{CENTER[2]}))))+exp(1)+20",
            )
        elif config.test_function == "rosenbrock":
            SCALE = 1
            minimum, maximum = find_min_max(rosenbrock, generate_grid(self.SIZE))
            self.evaluation_function = linear_scaling(
                rosenbrock,
                (0, 0, 0),
                minimum,
                maximum,
                to_string_mode=True,
                f=f"100*(({SCALE}*(y-{CENTER[1]})-({SCALE}*(x-{CENTER[0]}))^2)^2+({SCALE}*(z-{CENTER[2]})-({SCALE}*(y-{CENTER[1]}))^2)^2)+({SCALE}*(x-{CENTER[0]})-1)^2+({SCALE}*(y-{CENTER[1]})-1)^2",
            )
        elif config.test_function == "eggholder":
            SCALE = 1024
            minimum, maximum = find_min_max(eggholder, generate_grid(self.SIZE))
            self.evaluation_function = linear_scaling(
                eggholder,
                (0, 0, 0),
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(0-{SCALE}*(x-{CENTER[0]}))*sin(sqrt(abs({SCALE}*(x-{CENTER[0]})-{SCALE}*(y-{CENTER[1]})-47)))-({SCALE}*(y-{CENTER[1]})+47)*sin(sqrt(abs(0.5*{SCALE}*(x-{CENTER[0]})+{SCALE}*(y-{CENTER[1]})+47)))-{SCALE}*(y-{CENTER[1]})*sin(sqrt(abs({SCALE}*(y-{CENTER[1]})-{SCALE}*(z-{CENTER[2]})-47)))-({SCALE}*(z-{CENTER[2]})+47)*sin(sqrt(abs(0.5*{SCALE}*(y-{CENTER[1]})+{SCALE}*(z-{CENTER[2]})+47)))",
            )
        elif config.test_function == "rastrigin_mod":
            SCALE = 10
            minimum, maximum = find_min_max(rastrigin_mod, generate_grid(self.SIZE))
            PI = 3.14159
            self.evaluation_function = linear_scaling(
                rastrigin_mod,
                (0, 0, 0),
                minimum,
                maximum,
                to_string_mode=True,
                f=f"(({SCALE}*(x-{CENTER[0]}))^2-10*cos(2*{PI}*{SCALE}*(x-{CENTER[0]})))+(({SCALE}*(y-{CENTER[1]}))^2-10*cos(2*{PI}*{SCALE}*(y-{CENTER[1]})))+(({SCALE}*(z-{CENTER[2]}))^2-10*cos(2*{PI}*{SCALE}*(z-{CENTER[2]})))-({SCALE}*(x-{CENTER[0]}))*({SCALE}*(z-{CENTER[2]}))",
            )
        else:
            raise ValueError(f"Unknown function: {config.test_function}.")
        self.enable_gradient = enable_gradient
        self.output_name = (
            output_name if output_name is not None else self.DEFAULT_OUTPUT_MESH_NAME
        )

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

    def evaluate(self) -> None:
        if self.enable_gradient:
            cmd = f'precice-aste-evaluate -m {self.parameters.input_mesh} -f "{self.evaluation_function}" -d "{DEFAULT_DATA_NAME}" -o {self.DEFAULT_INPUT_MESH_NAME} --gradient --log DEBUG'
        else:
            cmd = f'precice-aste-evaluate -m {self.parameters.input_mesh} -f "{self.evaluation_function}" -d "{DEFAULT_DATA_NAME}" -o {self.DEFAULT_INPUT_MESH_NAME} --gradient --log DEBUG'
        self._run_command(cmd)

    def partition(
        self,
        mesh_a_name: str = DEFAULT_MESH_NAME_A,
        mesh_b_name: str = DEFAULT_MESH_NAME_B,
    ) -> None:
        cmd_a = f"precice-aste-partition -m {self.DEFAULT_INPUT_MESH_NAME} -n {self.nb_process} -o {mesh_a_name} --dir {mesh_a_name} --algorithm meshfree"
        cmd_b = f"precice-aste-partition -m {self.parameters.output_mesh} -n {self.nb_process} -o {mesh_b_name} --dir {mesh_b_name} --algorithm meshfree"
        self._run_command_parallel(cmd_a, cmd_b)

    def run(self):
        path_to_mapped = pathlib.Path("mapped/")
        path_to_mapped.mkdir(parents=True, exist_ok=True)
        cmd_a = f'mpirun -n {self.nb_process} precice-aste-run -p A --mesh {self.DEFAULT_MESH_NAME_A}/{self.DEFAULT_MESH_NAME_A} --data "{DEFAULT_DATA_NAME}"'
        cmd_b = f'mpirun -n {self.nb_process} precice-aste-run -p B --mesh {self.DEFAULT_MESH_NAME_B}/{self.DEFAULT_MESH_NAME_B} --output mapped/mapped --data "{DEFAULT_INTERPOLATED_DATA_NAME}"'
        self._run_command_parallel(cmd_a, cmd_b)

    def join(self):
        cmd = f"precice-aste-join -m mapped/mapped -o {self.output_name} --recovery {self.DEFAULT_MESH_NAME_B}/{self.DEFAULT_MESH_NAME_B}_recovery.json"
        self._run_command(cmd)

    def stats(self):
        cmd = f'precice-aste-evaluate -m {self.output_name} -f "{self.evaluation_function}" -d "Error" --diffdata "{DEFAULT_INTERPOLATED_DATA_NAME}" --diff --stats --log DEBUG'
        self._run_command(cmd)

    def _vtu_to_csv(self, path_to_file: pathlib.Path, data: str, path_to_output: str):
        reader = vtk.vtkXMLUnstructuredGridReader()  # type: ignore
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
        output_array = np.column_stack((x, y, z, data_values))
        np.savetxt(
            path_to_output,
            output_array,
            delimiter=",",
            header=f"x,y,z,{data}",
            comments="",
        )

    def save_results(self, folder_name: str, save_csv: bool = True):
        path_to_folder = PATH_TO_OUT / folder_name
        path_to_folder.mkdir(parents=True, exist_ok=True)
        # Read and process results
        if save_csv:
            self._vtu_to_csv(self.DEFAULT_INPUT_MESH_NAME, DEFAULT_DATA_NAME, str(path_to_folder / "input.csv"))
            self._vtu_to_csv(
                self.DEFAULT_OUTPUT_MESH_NAME, DEFAULT_INTERPOLATED_DATA_NAME, str(path_to_folder / "output.csv")
            )
        # Move all meshes with their respective data
        self.DEFAULT_INPUT_MESH_NAME.rename(
            path_to_folder / self.DEFAULT_INPUT_MESH_NAME.name
        )
        self.DEFAULT_OUTPUT_MESH_NAME.rename(
            path_to_folder / self.DEFAULT_OUTPUT_MESH_NAME.name
        )
        # Move all configuration files
        pathlib.Path("result.stats.json").rename(path_to_folder / "stats.json")
        pathlib.Path("precice-config.xml").rename(path_to_folder / "precice-config.xml")
        # Copy config.json to the folder
        shutil.copy(self.parameters.path_to_config, path_to_folder)

    def clean(self):
        self._run_command("make clean")


class Process:
    SIZE = 100
    # Function mapping dictionary
    FUNCTION_MAP = {
        "sphere": sphere,
        "drop_wave": drop_wave,
        "ackley": ackley,
        "rosenbrock": rosenbrock,
        "eggholder": eggholder,
        "rastrigin_mod": rastrigin_mod,
    }

    def __init__(
        self,
        input_csv_file: pathlib.Path,
        output_csv_file: pathlib.Path,
        function_name: str,
    ) -> None:
        self.input_csv_file = input_csv_file
        self.output_csv_file = output_csv_file
        # Get the actual function from the map
        if function_name not in self.FUNCTION_MAP:
            raise ValueError(f"Unknown function: {function_name}")
        self.function = self.FUNCTION_MAP[function_name]

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

    def compute_error_metrics(self):
        x_in, y_in, z_in, values_in = self.read_csv(which="in")
        x_out, y_out, z_out, values_out = self.read_csv(which="out")
        n = x_out.size
        minimum, maximum = find_min_max(self.function, generate_grid(self.SIZE))
        predicted = np.array(
            [
                linear_scaling(self.function, (x, y, z), minimum, maximum)
                for x, y, z in zip(x_out, y_out, z_out)
            ]
        )
        relative_errors_pointwise = 100 * np.abs((predicted - values_out) / predicted)
        linfty_global = np.max(np.abs(values_out - predicted))
        rmse = np.sqrt(1 / n * np.sum((values_out - predicted) ** 2))
        return (
            x_out,
            y_out,
            z_out,
            values_out,
            relative_errors_pointwise,
            linfty_global,
            rmse,
        )

    def plot_data(self, path_to_file: pathlib.Path):
        # Load data
        (
            x_out,
            _,
            z_out,
            output_data,
            relative_errors,
            linfty,
            rmse,
        ) = self.compute_error_metrics()
        print(f"L_infty={linfty}, RMSE(global)={rmse}")

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
        plt.savefig(path_to_file, dpi=300, bbox_inches="tight")
        plt.close()

def main(folder_name: str = "results") -> None:
    configuration = ConfigParser(pathlib.Path("config.json"))
    working_on_blade = is_blade(configuration.input_mesh)
    do_gradient = True if working_on_blade else False
    if working_on_blade:
        assert configuration.test_function in [
            "sphere",
            "rosenbrock",
            "rastrigin_mod",
        ], "Unsupported function for blade."
    xmleditor = XMLEditor(configuration)
    xmleditor.write()
    run = Run(configuration, enable_gradient=do_gradient, output_name="result.vtu")
    run.evaluate()
    run.partition()
    run.run()
    run.join()
    run.stats()
    run.save_results(folder_name, save_csv=(not working_on_blade))
    run.clean()
    if not working_on_blade:
        process = Process(
            input_csv_file=PATH_TO_OUT / folder_name / "input.csv",
            output_csv_file=PATH_TO_OUT / folder_name / "output.csv",
            function_name=configuration.test_function,
        )
        process.compute_error_metrics()
        process.plot_data(
            path_to_file=PATH_TO_OUT / folder_name / "result.png",
        )
    return None

if __name__ == "__main__":
    main(folder_name="test_results")