# Script to automatically run ASTE
##! Standard library imports
import pathlib
import subprocess
import shutil
import json
from typing import Optional

##! Third-party librairies
import numpy as np
import vtk
import pandas as pd

##! Constants
# * Paths to relevant directories
PATH_TO_MESHES: pathlib.Path = pathlib.Path(
    "./meshes"
)  # ? Path to the meshes directory (blades and lid-driven cavity)
assert PATH_TO_MESHES.exists(), "Path to meshes does not exist!"
PATH_TO_STRETCHED: pathlib.Path = pathlib.Path(
    "./stretched"
)  # ? Path to the stretched meshes (based on lid-driven cavity) directory
assert PATH_TO_STRETCHED.exists(), "Path to stretched meshes does not exist!"
PATH_TO_OUT: pathlib.Path = pathlib.Path("./out")  # ? Path to the output directory
assert PATH_TO_OUT.exists(), "Path to output does not exist!"
PATH_TO_TEMPLATES: pathlib.Path = pathlib.Path(
    "./templates"
)  # ? Path to the templates (precice-config.xml) directory
assert PATH_TO_TEMPLATES.exists(), "Path to templates does not exist!"
# * Constants
DEFAULT_DATA_NAME: str = "InputData"
DEFAULT_INTERPOLATED_DATA_NAME: str = "InterpolatedData"
DEFAULT_NB_PROCS: int = 4  # ? Default number of processes to use for ASTE
CENTER: tuple = (0.5, 0.0, 0.5)  # ? Center of the domain for the lid-driven cavity

##! Test functions


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


def eggholder(x: tuple) -> float:
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


##! Class ConfigParser + related utilty functions


def is_blade(path_to_file: str) -> bool:
    return (
        True if "vtk" in path_to_file else False
    )  # ? All VTK files are blades, LDC files are in .vtu format


class ConfigParser:
    @staticmethod
    def _parse_config(path_to_config: pathlib.Path) -> dict:
        with open(path_to_config, "r") as file:
            config = json.load(file)
        return config

    @staticmethod
    def _parse_parameters(config: dict, method: str) -> tuple[str,str] | None:
        if "rbf" not in method:
            return None  # * If the method is not RBF, no parameters are needed
        else:
            parameters = config.get("additional-config", {})
            basis_function = parameters.get("basis-function", None)
            param = parameters.get("param", 0.0)
            return basis_function, param

    def __init__(self, path_to_config: pathlib.Path) -> None:
        self.path_to_config = path_to_config
        assert self.path_to_config.exists(), "Path to config does not exist!"
        self.config: dict = self._parse_config(self.path_to_config)
        self.input_mesh: str = str(PATH_TO_MESHES / self.config.get("input-mesh", ""))
        if not isinstance(self.input_mesh, str) or not self.input_mesh:
            raise ValueError("input-mesh must be provided in the config and must be a string.")
        self.output_mesh: str = str(PATH_TO_MESHES / self.config.get("output-mesh", ""))
        if not isinstance(self.output_mesh, str) or not self.output_mesh:
            raise ValueError("output-mesh must be provided in the config and must be a string.")
        self.method: str = self.config.get("mapping-method", "")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("mapping-method must be provided in the config and must be a string.")
        self.parameters: tuple = self._parse_parameters(self.config, self.method) or ()
        self.test_function: str = self.config.get("test-function", "")
        self.is_blade: bool = is_blade(self.input_mesh)
        self.nb_procs: int = self.config.get("nb-procs", DEFAULT_NB_PROCS)
        return None

    def __str__(self) -> str:
        return f"ConfigParser({self.path_to_config})"

    def print_to_terminal(self) -> None:
        with open(self.path_to_config, "r") as file:
            json_object = json.load(file)
        print(json.dumps(json_object, indent=2))


##! Class XMLEditor + related utility functions 

class XMLEditor:
    STD_TEMPLATE: str = "no_rbf.txt"
    RBF_TEMPLATE: str = "rbf_std.txt"

    @staticmethod
    def _parse_xml_template(path_to_template: pathlib.Path) -> str:
        with open(path_to_template, "r") as file:
            xml_content = file.read()
        return xml_content
    
    def __init__(self, config: ConfigParser) -> None:
        self.config: ConfigParser = config
        self.template: str = (
            self.RBF_TEMPLATE if "rbf" in self.config.method else self.STD_TEMPLATE
        )
        self.path_to_template: pathlib.Path = PATH_TO_TEMPLATES / self.template
        assert self.path_to_template.exists(), "Path to template does not exist!"
        self.xml_content: str = self._parse_xml_template(self.path_to_template)
        return None
    
    def replace_placeholders(self) -> None:
        method = self.config.method
        self.xml_content = self.xml_content.replace("$method$", method)
        if "rbf" in method:
            basis_function, param = self.config.parameters
            print(param)
            if basis_function in ["multiquadrics", "inverse-multiquadrics", "gaussian"]:
                param_name = "shape-parameter"
                param_value = param
                self.xml_content = self.xml_content.replace("$basis-function$", basis_function)
                self.xml_content = self.xml_content.replace("$param$", param_name)
                self.xml_content = self.xml_content.replace("$value$", param_value)
            elif basis_function in ["compact-polynomial-c0", "compact-polynomial-c2", "compact-polynomial-c4", "compact-polynomial-c6", "compact-polynomial-c8", "compact-tps-c2"]:
                param_name = "support-radius"
                param_value = param
                self.xml_content = self.xml_content.replace("$basis-function$", basis_function)
                self.xml_content = self.xml_content.replace("$param$", param_name)
                self.xml_content = self.xml_content.replace("$value$", param_value)
            else:
                assert basis_function in ["thin-plate-splines", "volumes-splines"]
                self.xml_content = self.xml_content.replace("$basis-function$", basis_function)
                self.xml_content = self.xml_content.replace("$param$:$value$", "")

    def write_to_file(self) -> None:
        path_to_output = pathlib.Path(".") / "precice-config.xml"
        with open(path_to_output, "w") as file:
            file.write(self.xml_content)
        return None
    
##! Class Run + related utility functions

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
        self.nb_process = config.nb_procs
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
    
if __name__ == "__main__":
    # * Parse the config file
    path_to_config = pathlib.Path("./config.json")
    config = ConfigParser(path_to_config)
    config.print_to_terminal()

    # * Create the XML editor and replace placeholders
    xml_editor = XMLEditor(config)
    xml_editor.replace_placeholders()

    # * Write the XML content to a file
    xml_editor.write_to_file()

    # * Create the Run object
    run = Run(config, enable_gradient=True, dry_run=False)
    run.evaluate()
    run.partition()
    run.run()
    run.join()
    run.stats()
    run.save_results("results", save_csv=True)
    run.clean()