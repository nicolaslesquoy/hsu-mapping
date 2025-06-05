# Local libraries
# import run
# import mesh

# Standard libraries
# import argparse
import pathlib
import subprocess

# Third-party libraries
# import numpy as np
# import meshio

# Constants
PATH_TO_TEMPLATES = pathlib.Path(__file__).parent / "templates"
PATH_TO_MESHES = pathlib.Path(__file__).parent / "meshes"
PATH_TO_STRETCHED = pathlib.Path(__file__).parent / "stretched"


class MeshGenerator:
    """Class to handle mesh generation with GMSH and processing."""

    QUAD_TEMPLATE = PATH_TO_TEMPLATES / "quad.geo"
    TRI_TEMPLATE = PATH_TO_TEMPLATES / "tri.geo"

    def __init__(self, mesh_name, nb_elements: int = 100, element_type: str = "quad"):
        self.mesh_name = mesh_name
        self.nb_elements = nb_elements
        if element_type == "quad":
            self.template = self.QUAD_TEMPLATE
        elif element_type == "tri":
            self.template = self.TRI_TEMPLATE
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        self.mesh_path = PATH_TO_MESHES / f"{self.mesh_name}.msh"

    def edit_template(self):
        """Edit the GMSH template with provided parameters."""
        with open(self.template, "r") as file:
            content = file.read()

        content = content.replace("$nb_elements$", str(self.nb_elements))

        with open(PATH_TO_MESHES / f"{self.mesh_name}.geo", "w") as file:
            file.write(content)

    def generate_mesh(self):
        """Generate the mesh using GMSH."""
        self.edit_template()
        
        try:
            # Create 2D mesh using GMSH
            subprocess.run(
                [
                    "gmsh",
                    str(PATH_TO_MESHES / f"{self.mesh_name}.geo"),
                    "-2",  # 2D mesh
                    "-format", "msh2", 
                    "-o",
                    str(PATH_TO_MESHES / f"{self.mesh_name}.msh"),
                ],
                check=True,
            )
            
            # Verify the mesh file exists
            mesh_file = PATH_TO_MESHES / f"{self.mesh_name}.msh"
            if not mesh_file.exists() or mesh_file.stat().st_size == 0:
                raise FileNotFoundError(f"GMSH failed to create valid mesh file: {mesh_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"GMSH mesh generation failed: {e}")
            raise

    def convert_mesh(self):
        """Convert the mesh to a different format using meshio Python API."""
        import meshio
        
        print(f"Converting {self.mesh_path} to VTU format...")
        try:
            # Read the mesh
            mesh = meshio.read(str(self.mesh_path))
            
            # Write to VTU format
            vtu_path = str(PATH_TO_MESHES / f"{self.mesh_name}.vtu")
            meshio.write(vtu_path, mesh)
            
            print(f"Successfully converted to {vtu_path}")
        except Exception as e:
            print(f"Error converting mesh: {e}")
            raise

def batch_generate_meshes(element_type="quad"):
    """Generate multiple meshes in batch mode."""
    nb_elements: list = [2**i for i in range(5, 10)]
    for nb in nb_elements:
        mesh_name = f"gmsh_{element_type}_{nb}x{nb}"
        mesh_gen = MeshGenerator(mesh_name, nb_elements=nb, element_type=element_type)
        mesh_gen.generate_mesh()
        mesh_gen.convert_mesh()


class Batch:
    """Class to handle running simulations in batch mode."""
    pass


if __name__ == "__main__":
    # Example usage
    batch_generate_meshes(element_type="quad")
    batch_generate_meshes(element_type="tri")
    print("Mesh generation completed.")
