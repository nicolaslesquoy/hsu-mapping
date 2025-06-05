import vtk
import pathlib
import numpy as np

PATH_TO_MESHES = pathlib.Path(__file__).parent / "meshes"

def read_vtu(path_to_vtu: pathlib.Path):
    reader = vtk.vtkXMLUnstructuredGridReader()  # type: ignore
    reader.SetFileName(str(path_to_vtu))
    reader.Update()
    mesh = reader.GetOutput()
    
    # Get all field names
    point_data = mesh.GetPointData()
    num_arrays = point_data.GetNumberOfArrays()
    field_names = [point_data.GetArrayName(i) for i in range(num_arrays)]
    
    # Create dictionary of fields
    fields = {name: mesh.GetPointData().GetArray(name) 
             for name in field_names if name is not None}
    
    print(f"Fields in {path_to_vtu}: {list(fields.keys())}")
    return fields

if __name__ == "__main__":
    path_to_vtu = pathlib.Path(PATH_TO_MESHES / "gmsh_quad_32x32.vtu")
    read_vtu(path_to_vtu)