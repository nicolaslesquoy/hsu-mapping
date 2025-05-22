import pathlib

import vtk
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_STRETCHED = pathlib.Path("./stretched")
PATH_TO_MESHES = pathlib.Path("./meshes")


# def swap(i: int, j: int, subject: list) -> None:
#     e1 = subject[i]
#     e2 = subject[j]
#     subject[i] = e2
#     subject[j] = e1
#     return None


def load(path_to_mesh: pathlib.Path):
    reader = vtk.vtkXMLUnstructuredGridReader()  # type: ignore
    reader.SetFileName(str(path_to_mesh))  # Convert Path to string
    reader.Update()
    ugrid = reader.GetOutput()
    points = ugrid.GetPoints()
    extract = []  # x,y,z,id
    for i in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(i)
        extract.append([x, y, z, i])
    return extract, ugrid  # Return both points and original content


def filter_points(
    points: list, coordinate: str, target: float, epsilon: float = 1e-6
) -> list:
    coord_index = {"x": 0, "y": 1, "z": 2}[coordinate]
    return [p for p in points if abs(p[coord_index] - target) < epsilon]

def modify(points: list) -> None:
    bottom = filter_points(points, coordinate="z", target=min(p[2] for p in points))
    left = filter_points(points, coordinate="x", target=min(p[0] for p in points))
    middle_index = len(bottom) // 2
    L = (max(p[0] for p in points) - min(p[0] for p in points)) / 2
    s = 1.5
    dx = L * (1 - s) / (1 - s**(middle_index - 1))
    # Stretching vertically (z coordinate)
    for i in range(len(bottom)):
        column_pos = bottom[i][0]
        column_indices = [
            point[3] for point in points if abs(point[0] - column_pos) < 1e-6
        ]
        column = [points[idx] for idx in column_indices]
        column.sort(key=lambda p: p[2])  # Sort by z coordinate
        # Stretch lower half
        z0 = column[0][2]
        old_z = z0
        for j in range(1, middle_index):
            new_z = old_z + dx * s**(j - 1)
            points[column[j][3]][2] = new_z
            old_z = new_z
        # Reverse column
        column.reverse()
        # Stretch upper half
        zn = column[0][2]
        old_z = zn
        for j in range(1, middle_index):
            new_z = old_z - dx * s**(j - 1)
            points[column[j][3]][2] = new_z
            old_z = new_z
    # Stretching horizontally (x coordinate)
    for i in range(1, len(left)):
        row_pos = left[i][2]
        row_indices = [
            point[3] for point in points if abs(point[2] - row_pos) < 1e-6
        ]
        row = [points[idx] for idx in row_indices]
        row.sort(key=lambda p: p[0])
        # Stretch left half
        x0 = row[0][0]
        old_x = x0
        for j in range(1, middle_index):
            new_x = old_x + dx * s**(j - 1)
            points[row[j][3]][0] = new_x
            old_x = new_x
        # Reverse row
        row.reverse()
        # Stretch right half
        xn = row[0][0]
        old_x = xn
        for j in range(1, middle_index):
            new_x = old_x - dx * s**(j - 1)
            points[row[j][3]][0] = new_x
            old_x = new_x
    return None

def write(points: list, original_grid, filename: str):
    # Create a deep copy of the original grid to preserve all data
    new_grid = vtk.vtkUnstructuredGrid()  # type: ignore
    new_grid.DeepCopy(original_grid)

    # Only modify the points
    new_points = vtk.vtkPoints()  # type: ignore
    for point in points:
        new_points.InsertPoint(
            point[3], point[0], point[1], point[2]
        )  # Use original ID

    # Set the modified points in the grid
    new_grid.SetPoints(new_points)

    # Write the modified mesh to a file
    writer = vtk.vtkXMLUnstructuredGridWriter()  # type: ignore
    writer.SetFileName(str(PATH_TO_STRETCHED / filename))
    writer.SetInputData(new_grid)
    writer.Write()
    print(f"Mesh written to {filename}")


def process(path_to_mesh: pathlib.Path, filename: str) -> None:
    points, original_grid = load(path_to_mesh)
    modify(points)
    write(points, original_grid, filename)
    return None

def plot(path_to_file: pathlib.Path) -> None:
    # Plot mesh as a grid
    points, _ = load(path_to_file)
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 2])
    plt.savefig(f"{path_to_file.stem}.png")
    plt.close()

if __name__ == "__main__":
    process(
        PATH_TO_MESHES / "fluid_nodes_fastest_32.vtu", "fluid_nodes_fastest_32_stretched.vtu"
    )
    plot(PATH_TO_STRETCHED / "fluid_nodes_fastest_32_stretched.vtu")