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
    return [p for p in points if abs(p[coord_index] - target) < 1e-6]


def transform(points: list, reference: list, coordinate: str):
    running_coord = {"x": 0, "y": 1, "z": 2}[coordinate]
    print(running_coord)
    for i in range(len(reference)):
        pos = reference[i][running_coord]
        indices = [
            point[3] for point in points if abs(point[running_coord] - pos) < 1e-6
        ]
        collection = [points[idx] for idx in indices]
        collection.sort(key=lambda p: p[2])
        n = len(collection)
        if n < 2:
            continue
        x0 = collection[0][running_coord]
        
        if n % 2 == 0:
            middle = n // 2 - 1
        else:
            middle = n // 2
        L = collection[-middle - 1][2] - x0
        s = 1.2

        if middle <= 1 or s == 1:
            continue

        denominator = 1 - s ** (middle - 1)
        if denominator == 0:
            continue

        dx = L * (1 - s) / denominator

        # Direct modification of points list using indices
        for j in range(1, len(collection) - middle):
            idx = collection[j][3]  # Get original point index
            points[idx][running_coord] = x0 + dx * s**j

        # Second pass - reversed
        for j in range(1, len(collection) - middle):
            idx = collection[len(collection)-j-1][3]  # Get index from end
            points[idx][running_coord] = x0 + dx * s**j


def modify(points: list):
    target_x = min([p[0] for p in points])
    # target_z = min([p[2] for p in points])
    # print(target_x)
    reference_x = filter_points(points, coordinate="x", target=target_x)
    # print(reference_x)
    # reference_z = filter_points(points, coordinate="z", target=target_z)

    transform(points, reference=reference_x, coordinate="z")
    # transform(points, reference=reference_z, coordinate="x")


def write(points: list, original_grid: vtk.vtkUnstructuredGrid, filename: str):
    # Create a deep copy of the original grid to preserve all data
    new_grid = vtk.vtkUnstructuredGrid()
    new_grid.DeepCopy(original_grid)

    # Only modify the points
    new_points = vtk.vtkPoints()
    for point in points:
        new_points.InsertPoint(
            point[3], point[0], point[1], point[2]
        )  # Use original ID

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
    plt.scatter(points[:, 0], points[:, 2])
    plt.savefig(f"{path_to_file.stem}.png")
    plt.close()


if __name__ == "__main__":
    main()
    plot(pathlib.Path("input.vtu"))
    plot(pathlib.Path("input_modified.vtu"))
