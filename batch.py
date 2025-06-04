# Local libraries
import run
import mesh

# Standard libraries
import argparse
import pathlib
import subprocess

# Third-party libraries
import numpy as np
import meshio

# Constants
PATH_TO_TEMPLATES = pathlib.Path(__file__).parent / "templates"
PATH_TO_MESHES = pathlib.Path(__file__).parent / "meshes"
PATH_TO_STRETCHED = pathlib.Path(__file__).parent / "stretched"

class MeshGenerator:
    """Class to handle mesh generation with GMSH and processing."""
    pass

class Batch