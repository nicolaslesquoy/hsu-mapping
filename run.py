# Script to automatically run ASTE
##! Standard library imports
import pathlib
import subprocess
import shutil
import json

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


def is_blade(path_to_file: pathlib.Path) -> bool:
    return (
        True if "vtk" in path_to_file.name else False
    )  # ? All VTK files are blades, LDC files are in .vtu format


class ConfigParser:
    @staticmethod
    def _parse_config(path_to_config: pathlib.Path) -> dict:
        with open(path_to_config, "r") as file:
            config = json.load(file)
        return config

    @staticmethod
    def _parse_parameters(config: dict, method: str) -> list | None:
        if "rbf" not in method:
            return None  # * If the method is not RBF, no parameters are needed
        else:
            parameters = config.get("additional-config", {})
            assert parameters, (
                "No parameters found in the config file!"
            )  # Check if parameters are present
            if method != "rbf-pum-direct":
                basis_function = parameters.get("basis-function", None)
                param = parameters.get(
                    "param", None
                )  # * Shape parameter or support radius
                return [basis_function, param, None, None]
            else:  # ? RBF Partition of Unity Method
                basis_function = parameters.get("basis-function", None)
                param = parameters.get(
                    "param", None
                )  # * Shape parameter or support radius
                vertices_per_cluster = parameters.get("vertices-per-cluster", None)
                relative_overlap = parameters.get("relative-overlap", None)
                return [
                    basis_function,
                    param,
                    vertices_per_cluster,
                    relative_overlap,
                ]

    def __init__(self, path_to_config: pathlib.Path) -> None:
        self.path_to_config = path_to_config
        assert self.path_to_config.exists(), "Path to config does not exist!"
        self.config: dict = self._parse_config(self.path_to_config)
        self.input_mesh: str = self.config.get("input-mesh", None)
        self.output_mesh: str = self.config.get("output-mesh", None)
        self.method: str = self.config.get("mapping-method", None)
        self.parameters: list = self._parse_parameters(self.config, self.method)
        self.test_function: str = self.config.get("test-function", None)
        self.is_blade: bool = is_blade(self.input_mesh)
        self.nb_procs: int = self.config.get("nb-procs", DEFAULT_NB_PROCS)

    def __str__(self) -> str:
        return f"ConfigParser({self.path_to_config})"

    def print_to_terminal(self) -> None:
        with open(self.path_to_config, "r") as file:
            json_object = json.load(file)
        print(json.dumps(json_object, indent=2))


##! Class XMLEditor + related utility functions 
