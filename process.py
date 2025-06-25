## Standard librairies
import pathlib
import subprocess
import shutil

## Third-party librairies
# * pip install -r requirements.txt
import vtk
import numpy as np

PATH_TO_MESHES = pathlib.Path("./meshes")
PATH_TO_DATA = pathlib.Path("./cfd-data")
PATH_TO_SAMPLES = pathlib.Path("./samples")

path_to_structure_nodes = PATH_TO_DATA / "Structure_Nodes-FASTEST.dt2000_3.vtu"
assert path_to_structure_nodes.exists(), f"File {path_to_structure_nodes} does not exist."
path_to_fluid_centers = PATH_TO_DATA / "Fluid_Centers-FASTEST.dt2000_3.vtu"
assert path_to_fluid_centers.exists(), f"File {path_to_fluid_centers} does not exist."
path_to_fluid_nodes = PATH_TO_DATA / "Fluid_Nodes-FASTEST.dt2000_3.vtu"
assert path_to_fluid_nodes.exists(), f"File {path_to_fluid_nodes} does not exist."

def vtu_to_csv(path_to_file: pathlib.Path, data: str, path_to_output: str):
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
        data_values = np.array(data_array)
        assert data_values is not None, f"Data {data} is None in the VTU file."
        dim = data_values[0].shape[0]
        x = point_coords[:, 0]
        y = point_coords[:, 1]
        z = point_coords[:, 2]
        if dim == 1:
            output_array = np.column_stack((x, y, z, data_values))
            field_name = "scalar"
        if dim != 1:
            vx, vy, vz = data_values[:, 0], data_values[:, 1], data_values[:, 2]
            magnitude = np.sqrt(vx**2 + vy**2 + vz**2) # Calculate the magnitude
            output_array = np.column_stack((x, y, z, vx, vy, vz, magnitude))
            field_name = "vx,vy,vz,magnitude"
        # Write to file
        np.savetxt(
            path_to_output,
            output_array, # type: ignore
            delimiter=",",
            header=f"x,y,z,{field_name}", # type: ignore
            comments="",
        )

if __name__ == "__main__":
    # Read the centers nodes
    # reader = vtk.vtkXMLUnstructuredGridReader()  # type: ignore
    # reader.SetFileName(str(path_to_fluid_centers))
    # reader.Update()
    # mesh = reader.GetOutput()
    # points = mesh.GetPoints()
    # point_coords = np.array(
    #     [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    # )
    # data_array = mesh.GetPointData().GetArray("Forces")
    # print(np.array(data_array)[0])


    # Convert the structure nodes
    vtu_to_csv(
        path_to_file=path_to_fluid_centers,
        data="Forces",
        path_to_output=str("structure_nodes.csv"),
    )