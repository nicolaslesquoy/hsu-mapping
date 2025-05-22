import vtk
import pathlib

import numpy as np
import matplotlib.pyplot as plt

PATH_TO_STRETCHED = pathlib.Path("./stretched")


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


def modify(points: list):
    # Process columns (vertical stretching)
    bottom = filter_points(points, coordinate="z", target=0.015625)
    for i in range(len(bottom)):
        column_pos = bottom[i][0]
        # Get indices of points in this column
        column_indices = [
            point[3] for point in points if abs(point[0] - column_pos) < 1e-6
        ]
        column = [points[idx] for idx in column_indices]
        column.sort(key=lambda p: p[2])  # Sort by z coordinate

        n = len(column)
        if n < 2:
            continue

        is_odd = n % 2 != 0
        middle = n // 2 if is_odd else n // 2 - 1

        z0 = column[0][2]
        zn = column[-1][2]
        s = 1.5

        # Avoid division by zero or negative power issues
        if middle <= 1 or s == 1:
            continue

        # Lower half stretching
        L_lower = column[middle][2] - z0
        denominator_lower = 1 - s ** (middle - 1)
        if denominator_lower != 0:
            dx_lower = L_lower * (1 - s) / denominator_lower
            for j in range(1, middle):
                new_z = z0 + dx_lower * s**j
                points[column[j][3]][2] = new_z

        # Upper half stretching
        L_upper = zn - column[middle][2]
        start_idx = middle + 1 if is_odd else middle + 1
        denominator_upper = 1 - s ** (n - start_idx - 1)
        if denominator_upper != 0:
            dx_upper = L_upper * (1 - s) / denominator_upper
            for j in range(start_idx, n - 1):
                new_z = zn - dx_upper * s ** (n - j - 1)
                points[column[j][3]][2] = new_z

    # Process rows (horizontal stretching)
    left = filter_points(points, coordinate="x", target=min(p[0] for p in points))
    for i in range(len(left)):
        row_pos = left[i][2]
        row_indices = [point[3] for point in points if abs(point[2] - row_pos) < 1e-6]
        row = [points[idx] for idx in row_indices]
        row.sort(key=lambda p: p[0])

        n = len(row)
        if n < 2:
            continue

        is_odd = n % 2 != 0
        middle = n // 2 if is_odd else n // 2 - 1

        x0 = row[0][0]
        xn = row[-1][0]
        s = 1.5

        if middle <= 1 or s == 1:
            continue

        # Left half stretching
        L_left = row[middle][0] - x0
        denominator_left = 1 - s ** (middle - 1)
        if denominator_left != 0:
            dx_left = L_left * (1 - s) / denominator_left
            for j in range(1, middle):
                new_x = x0 + dx_left * s**j
                points[row[j][3]][0] = new_x

        # Right half stretching
        L_right = xn - row[middle][0]
        start_idx = middle + 1 if is_odd else middle + 1
        denominator_right = 1 - s ** (n - start_idx - 1)
        if denominator_right != 0:
            dx_right = L_right * (1 - s) / denominator_right
            for j in range(start_idx, n - 1):
                new_x = xn - dx_right * s ** (n - j - 1)
                points[row[j][3]][0] = new_x


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
    plt.savefig(f"{path_to_file.name}.png")
    plt.close()


if __name__ == "__main__":
    main()
    plot(pathlib.Path("input.vtu"))
    plot(pathlib.Path("input_modified.vtu"))
