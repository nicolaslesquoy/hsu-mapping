import vtk
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def swap(i: int, j: int, l: list) -> None:
    e1 = l[i]
    e2 = l[j]
    l[i] = e2
    l[j] = e1


def load(path_to_mesh: pathlib.Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path_to_mesh))  # Convert Path to string
    reader.Update()
    ugrid = reader.GetOutput()
    points = ugrid.GetPoints()
    extract = []  # x,y,z,id
    for i in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(i)
        extract.append([x, y, z, i])
    return extract, ugrid  # Return both points and original grid

def filter_points(points, coordinate, target):
    coord_index = {"x": 0, "y": 1, "z": 2}[coordinate]
    return [p for p in points if p[coord_index] == target]


def get_column(points, x_val):
    return sorted([p for p in points if p[0] == x_val], key=lambda p: p[2])


def modify(points: list):
    bottom = filter_points(points, coordinate="z", target=0)
    for i in range(1, len(bottom)):
        column_pos = bottom[i][0]
        column = get_column(points, column_pos)
        n = len(column)

        if n < 2:
            continue

        z0 = column[0][2]

        if n % 2 == 0:
            middle = n // 2 - 1
        else:
            middle = n // 2

        zlast = column[middle][2]
        L = column[-middle - 1][2] - z0
        s = 1.2

        # Avoid division by zero or negative power issues
        if middle <= 1 or s == 1:
            continue

        denominator = 1 - s ** (middle - 1)
        if denominator == 0:
            continue

        dx = L * (1 - s) / denominator

        for j in range(1, len(column) - middle):
            column[j][2] = z0 + dx * s**j

        for i in range(len(column)):
            index = column[i][3]
            points[index][2] = column[i][2]
    
def write(points: list, original_grid: vtk.vtkUnstructuredGrid, filename: str):
    # Create a deep copy of the original grid to preserve all data
    new_grid = vtk.vtkUnstructuredGrid()
    new_grid.DeepCopy(original_grid)
    
    # Only modify the points
    new_points = vtk.vtkPoints()
    for point in points:
        new_points.InsertPoint(point[3], point[0], point[1], point[2])  # Use original ID
    
    # Set the modified points in the grid
    new_grid.SetPoints(new_points)

    # Write the modified mesh to a file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(new_grid)
    writer.Write()
    print(f"Mesh written to {filename}")



def main():
    # Input and output paths
    input_mesh = pathlib.Path("input.vtu")
    output_mesh = pathlib.Path("input_modified.vtu")

    # Load the mesh
    points, original_grid = load(input_mesh)
    
    # Modify point coordinates
    modify(points)
    
    # Write modified mesh
    write(points, original_grid, str(output_mesh))

def plot(path_to_file: pathlib.Path):
    points, _ = load(path_to_file)
    points = np.array(points)
    plt.scatter(points[:,0], points[:,2])
    plt.savefig(f"{path_to_file.name}.png")
    plt.close()

if __name__ == "__main__":
    main()
    plot(pathlib.Path("input.vtu"))
    plot(pathlib.Path("input_modified.vtu"))