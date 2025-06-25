## Standard librairies
import pathlib
import subprocess
import shutil
import json
from typing import Optional
from time import sleep

## Third-party librairies
# * pip install -r requirements.txt
import vtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import random
import pandas as pd

p = 1 / 2.54

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## FUNCTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ? Center of the computational domain for lid-driven cavity
CENTER = (0.5, 0.0, 0.5)
# CENTER = (0.5, 0.5, 0.0)  # Center for GMSH mesh generation


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


def parse_parameters(method: str, parameters: dict) -> tuple[str, float, float] | None:
    if "rbf" not in method:
        return None
    else:
        basis_function: str = parameters["basis-function"]
        support_radius: float = float(parameters["support-radius"])
        shape_parameter: float = float(parameters["shape-parameter"])
    return (basis_function, support_radius, shape_parameter)


class ConfigParser:
    """
    This class parses a JSON config file for a given case.
    """

    def __init__(self, path_to_config: pathlib.Path) -> None:
        assert path_to_config.exists(), "Configuration file not found."
        config = read_json(path_to_config)
        self.path_to_config: pathlib.Path = path_to_config
        if "stretched" in config["input-mesh"]:
            self.input_mesh: pathlib.Path = PATH_TO_STRETCHED / config["input-mesh"]
        else:
            self.input_mesh: pathlib.Path = PATH_TO_MESHES / config["input-mesh"]
        assert self.input_mesh.exists(), "Input mesh not found."
        if "stretched" in config["output-mesh"]:
            self.output_mesh: pathlib.Path = PATH_TO_STRETCHED / config["output-mesh"]
        else:
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
        self.additional_parameters: tuple[str, float, float] | None = parse_parameters(
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
        # Get the additional parameters for RBF methods
        if config.additional_parameters is not None:
            self.additional_parameters: tuple[str, float, float] = (
                config.additional_parameters
            )
        # Ajust the support radius if needed
        # suggested_radius = adjust_support_radius(config.input_mesh)
        # if (
        #     self.additional_parameters is not None
        #     and self.additional_parameters[1] > suggested_radius
        #     and suggested_radius > 0.1
        # ):
        #     print("Replacing support radius with suggested value.")
        #     self.additional_parameters = (
        #         self.additional_parameters[0],
        #         suggested_radius,
        #         self.additional_parameters[2],
        #     )
        # Get the template file to create the XML config
        if "rbf" in self.mapping_method and self.additional_parameters[0] in [
            "thin-plate-splines",
            "volume-splines",
        ]:
            # Global support and no shape parameter
            self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / "rbf_global.txt"
        elif "rbf" in self.mapping_method and self.additional_parameters[0] in [
            "multiquadrics",
            "inverse-multiquadrics",
        ]:
            # Global support and shape parameter
            self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / "rbf_shape.txt"
        elif "rbf" in self.mapping_method and self.additional_parameters[0] in [
            "gaussian"
        ]:
            # Local support and shape parameter
            self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / "rbf_gaussian.txt"
        elif "rbf" in self.mapping_method and self.additional_parameters[0] not in [
            "thin-plate-spline",
            "gaussian",
            "multiquadrics",
            "volume-splines",
            "inverse-multiquadrics",
        ]:
            # Local support and no shape parameter
            self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / "rbf_local.txt"
        else:
            # No RBF
            self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / "no_rbf.txt"

    def edit(self) -> str:
        lines = read_txt(self.path_to_template)
        content: str = "".join(lines)

        if "rbf" in self.mapping_method and self.additional_parameters[0] in [
            "thin-plate-splines",
            "volume-splines",
        ]:
            # Global support and no shape parameter
            content = content.replace("$rbf_type$", self.mapping_method)  # Method name
            content = content.replace(
                "$basis-function$", str(self.additional_parameters[0])
            )  # Basis function
        elif "rbf" in self.mapping_method and self.additional_parameters[0] in [
            "multiquadrics",
            "inverse-multiquadrics",
        ]:
            # Global support and shape parameter
            content = content.replace("$rbf_type$", self.mapping_method)  # Method name
            content = content.replace(
                "$basis-function$", str(self.additional_parameters[0])
            )  # Basis function
            content = content.replace(
                "$shape-parameter$", str(self.additional_parameters[2])
            )  # Shape parameter
        elif "rbf" in self.mapping_method and self.additional_parameters[0] in [
            "gaussian"
        ]:
            # Local support and shape parameter
            content = content.replace("$rbf_type$", self.mapping_method)  # Method name
            content = content.replace(
                "$basis-function$", str(self.additional_parameters[0])
            )  # Basis function
            content = content.replace(
                "$radius$", str(self.additional_parameters[1])
            )  # Support radius
            content = content.replace(
                "$shape-parameter$", str(self.additional_parameters[2])
            )  # Shape parameter
        elif "rbf" in self.mapping_method and self.additional_parameters[0] not in [
            "thin-plate-spline",
            "gaussian",
            "multiquadrics",
            "volume-splines",
            "inverse-multiquadrics",
        ]:
            # Local support and no shape parameter
            content = content.replace("$rbf_type$", self.mapping_method)  # Method name
            content = content.replace(
                "$basis-function$", str(self.additional_parameters[0])
            )  # Basis function
            content = content.replace(
                "$radius$", str(self.additional_parameters[1])
            )  # Support radius
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
    SIZE = 100

    def __init__(
        self,
        config: "ConfigParser",
        enable_gradient: bool = False,
        output_name: Optional[str] = None,
        dry_run: bool = False,
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
        self.dry_run = dry_run

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
            cmd = f'precice-aste-evaluate -m {self.parameters.input_mesh} -f "{self.evaluation_function}" -d "{DEFAULT_DATA_NAME}" -o {self.DEFAULT_INPUT_MESH_NAME} --log DEBUG'
        self._run_command(cmd, dry_run=self.dry_run)

    def partition(
        self,
        mesh_a_name: str = DEFAULT_MESH_NAME_A,
        mesh_b_name: str = DEFAULT_MESH_NAME_B,
    ) -> None:
        cmd_a = f"precice-aste-partition -m {self.DEFAULT_INPUT_MESH_NAME} -n {self.nb_process} -o {mesh_a_name} --dir {mesh_a_name} --algorithm meshfree"
        cmd_b = f"precice-aste-partition -m {self.parameters.output_mesh} -n {self.nb_process} -o {mesh_b_name} --dir {mesh_b_name} --algorithm meshfree"
        self._run_command_parallel(cmd_a, cmd_b, dry_run=self.dry_run)

    def run(self):
        path_to_mapped = pathlib.Path("mapped/")
        path_to_mapped.mkdir(parents=True, exist_ok=True)
        cmd_a = f'mpirun -n {self.nb_process} precice-aste-run -p A --mesh {self.DEFAULT_MESH_NAME_A}/{self.DEFAULT_MESH_NAME_A} --data "{DEFAULT_DATA_NAME}"'
        cmd_b = f'mpirun -n {self.nb_process} precice-aste-run -p B --mesh {self.DEFAULT_MESH_NAME_B}/{self.DEFAULT_MESH_NAME_B} --output mapped/mapped --data "{DEFAULT_INTERPOLATED_DATA_NAME}"'
        self._run_command_parallel(cmd_a, cmd_b, self.dry_run)

    def join(self):
        cmd = f"precice-aste-join -m mapped/mapped -o {self.output_name} --recovery {self.DEFAULT_MESH_NAME_B}/{self.DEFAULT_MESH_NAME_B}_recovery.json"
        self._run_command(cmd, self.dry_run)

    def stats(self):
        cmd = f'precice-aste-evaluate -m {self.output_name} -f "{self.evaluation_function}" -d "Error" --diffdata "{DEFAULT_INTERPOLATED_DATA_NAME}" --diff --stats --log DEBUG'
        self._run_command(cmd, self.dry_run)

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
            self._vtu_to_csv(
                self.DEFAULT_INPUT_MESH_NAME,
                DEFAULT_DATA_NAME,
                str(path_to_folder / "input.csv"),
            )
            self._vtu_to_csv(
                self.DEFAULT_OUTPUT_MESH_NAME,
                DEFAULT_INTERPOLATED_DATA_NAME,
                str(path_to_folder / "output.csv"),
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

        # Adjust layout for better appearance
        plt.tight_layout()
        plt.savefig(path_to_file, dpi=300, bbox_inches="tight")
        plt.close()


def main(folder_name: str = "results") -> None:
    configuration = ConfigParser(pathlib.Path("config.json"))
    working_on_blade = is_blade(configuration.input_mesh)
    do_gradient = (
        True if configuration.mapping_method == "nearest-neighbor-gradient" else False
    )
    if do_gradient:
        assert configuration.test_function in [
            "sphere",
            "rosenbrock",
            "rastrigin_mod",
        ], "Unsupported function for blade with gradient computation."
    xmleditor = XMLEditor(configuration)
    xmleditor.write()
    run = Run(
        configuration,
        enable_gradient=do_gradient,
        output_name="result.vtu",
        dry_run=False,
    )
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


def read_csv(path_to_csv: pathlib.Path):
    x, y, z, interpolated_data = np.loadtxt(
        path_to_csv, delimiter=",", skiprows=1, unpack=True
    )
    return (x, y, z, interpolated_data)


def compute_errors_without_borders(
    path_to_csv: pathlib.Path, function
) -> tuple[float, float, float]:
    x_out, y_out, z_out, values_out = read_csv(path_to_csv)
    # Compute the function values at the points
    minimum, maximum = find_min_max(function, generate_grid(Process.SIZE))
    predicted = np.array(
        [
            linear_scaling(function, (x, y, z), minimum, maximum)
            for x, y, z in zip(x_out, y_out, z_out)
        ]
    )
    values_out = np.array(values_out)
    x_min = np.min(x_out)
    x_max = np.max(x_out)
    z_min = np.min(z_out)
    z_max = np.max(z_out)
    # Remove all border points
    # All points with abs(x - x_min) < threshold, abs(x - x_max) < threshold, same for z
    threshold = 1e-5
    mask = (
        (np.abs(x_out - x_min) >= threshold)
        & (np.abs(x_out - x_max) >= threshold)
        & (np.abs(z_out - z_min) >= threshold)
        & (np.abs(z_out - z_max) >= threshold)
    )
    x_out = x_out[mask]
    y_out = y_out[mask]
    z_out = z_out[mask]
    values_out = values_out[mask]
    predicted = predicted[mask]
    n = x_out.size
    # Compute the error
    linfty_global = np.max(np.abs(values_out - predicted))
    rmse = np.sqrt(1 / n * np.sum((values_out - predicted) ** 2))
    median_error = np.median(np.abs(values_out - predicted))
    return float(rmse), float(linfty_global), float(median_error)


def compute_errors(path_to_dir: pathlib.Path, function):
    # iterate over all subdirectories
    errors = {}
    for subdir in path_to_dir.iterdir():
        if subdir.is_dir():
            name = subdir.name
            case_name = name.split("_")[0]
            csv_file = subdir / "output.csv"
            assert csv_file.exists(), f"CSV file not found in {subdir}"
            rmse, linfty, median_error = compute_errors_without_borders(
                csv_file, function
            )
            # print(
            #     f"Case: {case_name}, Mesh Size: {name.split('_')[1]}, RMSE: {rmse}, Linfty: {linfty}"
            # )
            mesh_size = name.split("_")[1]
            if case_name not in errors:
                errors[case_name] = []
            errors[case_name].append([mesh_size, rmse, linfty, median_error])
    return errors


def read_errors(path_to_dir: pathlib.Path, function):
    # Read errors from all subdirectories
    errors = {}
    fname = "stats.json"
    for subdir in path_to_dir.iterdir():
        if subdir.is_dir():
            name = subdir.name
            case_name = name.split("_")[0]
            stats_file = subdir / fname
            if not stats_file.exists():
                print(f"Stats file not found in {subdir}")
                continue
            with open(stats_file, "r") as f:
                data = json.load(f)
            rmse = data["relative-l2"]
            linfty = data["abs_max"]
            mesh_size = name.split("_")[1]
            if case_name not in errors:
                errors[case_name] = []
            errors[case_name].append([mesh_size, rmse, linfty])
    return errors


def plot_errors(errors: dict, function):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define mesh size range for reference lines
    mesh_sizes = np.array([32, 64, 128, 256])
    all_sizes = []
    all_linfties = []
    for case, error_data in errors.items():
        sizes = [float(data[0]) for data in error_data]
        linfties = [data[2] for data in error_data]
        all_sizes.extend(sizes)
        all_linfties.extend(linfties)
        markers = [
            "o",
            "s",
            "D",
            "^",
            "v",
            "<",
            ">",
            "p",
            "*",
            "h",
            "H",
            "+",
            "x",
            "|",
            "_",
        ]
        random_marker = random.choice(markers)
        ax.plot(sizes, linfties, marker=random_marker, label=case)

    ax.set_xlabel("Mesh Size", fontsize=12)
    ax.set_ylabel("$L_\\infty$ Error", fontsize=12)
    ax.set_title(f"Error Convergence Analysis for {function.__name__}", fontsize=14)
    ax.set_xscale("log")  # Use base-2 for clearer mesh size reading
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # Add legend outside the plot below the x-axis
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

    fig.tight_layout()
    plt.savefig(f"error_analysis_{function.__name__}.png", dpi=300, bbox_inches="tight")
    plt.close()


def batch(function):
    available_mapping_methods = {
        # "nearest-neighbor": "NN",
        # "nearest-projection": "NP",
        # "nearest-neighbor-gradient": "NNG",
        # "rbf/compact-polynomial-c0": "RBFC0",
        # "rbf/compact-polynomial-c2": "RBFC2",
        # "rbf/compact-polynomial-c4": "RBFC4",
        # "rbf/compact-polynomial-c6": "RBFC6",
        # "rbf/compact-polynomial-c8": "RBFC8",
        # "rbf/compact-tps-c2": "RBFTPSC2",
        "rbf/multiquadrics": "RBFMQ",
        "rbf/inverse-multiquadrics": "RBFIMQ",
        "rbf/gaussian": "RBFGAUSS",
        # "rbf/volume-splines": "RBFVS",
        # "rbf/thin-plate-splines": "RBFTPS",
    }
    mesh_sizes = ["32", "64", "128", "256"]
    input_meshes = [f"fluid_nodes_fastest_{size}.vtu" for size in mesh_sizes]
    output_meshes = [f"fluid_centers_fastest_{size}.vtu" for size in mesh_sizes]
    support_radius = [0.30, 0.15, 0.07, 0.04]
    for mapping_method, short_name in available_mapping_methods.items():
        for i in range(len(input_meshes)):
            input_mesh = input_meshes[i]
            output_mesh = output_meshes[i]
            # Create a config file for each case
            if "RBF" in short_name:
                config_data = {
                    "input-mesh": input_mesh,
                    "output-mesh": output_mesh,
                    "test-function": "rastrigin_mod",
                    "mapping-method": mapping_method.split("/")[0],
                    "additional-config": {
                        "basis-function": mapping_method.split("/")[1],
                        "support-radius": support_radius[i],
                        "shape-parameter": 0.09,
                    },
                    "nb-procs": 8,
                }
            else:
                config_data = {
                    "input-mesh": input_mesh,
                    "output-mesh": output_mesh,
                    "test-function": "rastrigin_mod",
                    "mapping-method": mapping_method,
                    "additional-config": {
                        "basis-function": None,
                        "support-radius": support_radius[i],
                        "shape-parameter": 0.09,
                    },
                    "nb-procs": 8,
                }
            config_file = pathlib.Path(f"config.json")
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            # Run the main function with the generated config file
            try:
                main(
                    folder_name=f"{short_name}_{output_mesh.split('.')[0].split('_')[-1]}_{function.__name__}"
                )
            except Exception as e:
                print(f"Error occurred for {short_name} with mesh {output_mesh}: {e}")
                continue


if __name__ == "__main__":
    # main(folder_name="test_rastrigin_mod")
    folder_name = "test"
    input_mesh = "0_0005.vtk"
    mapping_methods = [
        "nearest-neighbor",
        "nearest-projection",
        "nearest-neighbor-gradient",
        "rbf/compact-polynomial-c0",
        "rbf/compact-polynomial-c2",
        "rbf/compact-polynomial-c4",
        "rbf/compact-polynomial-c6",
        "rbf/compact-polynomial-c8",
        "rbf/compact-tps-c2",
        "rbf/multiquadrics",
        "rbf/inverse-multiquadrics",
        "rbf/gaussian",
        "rbf/volume-splines",
        "rbf/thin-plate-splines"
    ]
    output_meshes = list(PATH_TO_MESHES.glob("*.vtk"))
    # output_meshes = [
    #     "0_01.vtk",
    #     "0_001.vtk",
    #     "0_004.vtk",
    #     "0_009.vtk",
    #     "0_0007.vtk"
    # ]
    nb_vertex = 10
    config_data = {
        "input-mesh": str(input_mesh),
        "output-mesh": None,
        "test-function": "rastrigin_mod",
        "mapping-method": None,
        "additional-config": {
            "basis-function": None,
            "support-radius": None,
            "shape-parameter": None,
        },
        "nb-procs": 8,
    }
    safety_coef = 3 / 5
    results = {f"{method}": [] for method in mapping_methods}
    for method in mapping_methods:
        config_data["mapping-method"] = "rbf-pum-direct" if "rbf" in method else method
        for mesh in output_meshes:
            h = float(str(mesh.name).split(".")[0].replace("_","."))
            if "rbf" in method and method.split("/")[1] not in ["gaussian"]:
                config_data["additional-config"]["basis-function"] = method.split("/")[1]
                config_data["additional-config"]["support-radius"] = nb_vertex * h * safety_coef
                config_data["additional-config"]["shape-parameter"] = nb_vertex * h * safety_coef
            elif "rbf" in method and method.split("/")[1] in ["gaussian"]:
                config_data["additional-config"]["basis-function"] = method.split("/")[1]
                config_data["additional-config"]["support-radius"] = np.nan
                config_data["additional-config"]["shape-parameter"] = nb_vertex * h * safety_coef
            else:
                config_data["additional-config"]["basis-function"] = None
                config_data["additional-config"]["support-radius"] = nb_vertex * h * safety_coef
                config_data["additional-config"]["shape-parameter"] = nb_vertex * h * safety_coef
            # get file name with extension
            config_data["output-mesh"] = str(mesh.name)
            config_file = pathlib.Path(f"config.json")
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            try:
                main(folder_name=folder_name)
                # open stats.json
                with open(
                    PATH_TO_OUT / folder_name / "stats.json", "r"
                ) as f:
                    data = json.load(f)
                rmse = data["relative-l2"]
                linfty = data["abs_max"]
                results[method].append((h, rmse, linfty))
            except Exception as e:
                print(
                    f"Error occurred for {method} with mesh {mesh}: {e}. Skipping this case."
                )
                results[method].append((h, -1, -1))
                continue
    # Save results to a CSV file
    results_file = "results.csv"
    with open(results_file, "w") as f:
        f.write("Method,Mesh Size,RMSE,L_infty\n")
        for method, values in results.items():
            for h, rmse, linfty in values:
                f.write(f"{method},{h},{rmse},{linfty}\n")
    # with open("results.csv", "r") as f:
    #     lines = f.readlines()
    # results = {}
    # for line in lines[1:]:
    #     method, h, rmse, linfty = line.strip().split(",")
    #     if method not in results:
    #         results[method] = []
    #     results[method].append((float(h), float(rmse), float(linfty)))
    # # Plot the errors
    # for method, values in results.items():
    #     sizes = [v[0] for v in values]
    #     rmses = [v[1] for v in values]
    #     linfties = [v[2] for v in values]
    #     # Remove all occurrences of -1
    #     sizes = [s for s, r in zip(sizes, rmses) if r != -1]
    #     rmses = [r for r in rmses if r != -1]
    #     linfties = [l for l in linfties if l != -1]
    #     plt.scatter(sizes, rmses, label=f"{method} RMSE")
    #     # plt.plot(sizes, linfties, label=f"{method} Linfty", linestyle="--")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.savefig("error_analysis.png", dpi=300, bbox_inches="tight")
