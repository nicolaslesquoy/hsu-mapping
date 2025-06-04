import pathlib

import vtk
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

PATH_TO_STRETCHED = pathlib.Path("./stretched")
PATH_TO_MESHES = pathlib.Path("./meshes")


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
    # Replace linear search with NumPy vectorized operations
    points_array = np.array(points)
    coord_index = {"x": 0, "y": 1, "z": 2}[coordinate]
    mask = np.abs(points_array[:, coord_index] - target) < epsilon
    return points_array[mask].tolist()


def modify_nodes(nodes: list) -> list:
    # Create a deep copy of the original nodes to preserve all data
    new_nodes = [list(point) for point in nodes]  # Deep copy
    # Stretching the mesh (nodes only)
    bottom = filter_points(nodes, coordinate="z", target=min(p[2] for p in nodes))
    left = filter_points(nodes, coordinate="x", target=min(p[0] for p in nodes))
    middle_index = len(bottom) // 2 + 1
    L = (max(p[0] for p in nodes) - min(p[0] for p in nodes)) / 2
    s = 1.05
    if abs(s - 1.0) < 1e-6:
        dx = L / (middle_index - 1)
    else:
        dx = L * (1 - s) / (1 - s ** (middle_index - 1))
    # Stretching vertically (z coordinate)
    for i in range(len(bottom)):
        column_pos = bottom[i][0]
        column_indices = [
            point[3] for point in nodes if abs(point[0] - column_pos) < 1e-6
        ]
        column = [nodes[idx] for idx in column_indices]
        column.sort(key=lambda p: p[2])  # Sort by z coordinate
        # Stretch lower half
        z0 = column[0][2]
        old_z = z0
        for j in range(1, middle_index):
            new_z = old_z + dx * s ** (j - 1)
            new_nodes[column[j][3]][2] = new_z
            old_z = new_z
        # Reverse column
        column.reverse()
        # Stretch upper half
        zn = column[0][2]
        old_z = zn
        for j in range(1, middle_index):
            new_z = old_z - dx * s ** (j - 1)
            new_nodes[column[j][3]][2] = new_z
            old_z = new_z
    # Stretching horizontally (x coordinate)
    for i in range(len(left)):
        row_pos = left[i][2]
        row_indices = [point[3] for point in nodes if abs(point[2] - row_pos) < 1e-6]
        row = [nodes[idx] for idx in row_indices]
        row.sort(key=lambda p: p[0])
        # Stretch left half
        x0 = row[0][0]
        old_x = x0
        for j in range(1, middle_index):
            new_x = old_x + dx * s ** (j - 1)
            new_nodes[row[j][3]][0] = new_x
            old_x = new_x
        # Reverse row
        row.reverse()
        # Stretch right half
        xn = row[0][0]
        old_x = xn
        for j in range(1, middle_index):
            new_x = old_x - dx * s ** (j - 1)
            new_nodes[row[j][3]][0] = new_x
            old_x = new_x
    return new_nodes


def search_point(points: list, point: list) -> int:
    for i, p in enumerate(points):
        if all(abs(p[j] - point[j]) < 1e-6 for j in range(3)):
            return p[3]  # Return the index of the point
    return -1


def dist(point1: list, point2: list) -> float:
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[2] - point2[2]) ** 2)


def modify_centers(old_centers: list, old_nodes: list, new_nodes: list) -> list:
    new_centers = [list(point) for point in old_centers]  # Deep copy
    old_nodes_array = np.array(old_nodes)
    # Build KD-tree for fast nearest neighbor search
    tree = KDTree(old_nodes_array[:, [0, 2]])
    for i, center in enumerate(old_centers):
        # Query KDTree for k nearest neighbors
        _, indices = tree.query([center[0], center[2]], k=4)
        # Get indices of nearest nodes
        indices_list = indices.tolist() if hasattr(indices, "tolist") else [indices]
        corners = [old_nodes[idx][3] for idx in indices_list]  # type: ignore
        new_corners = [new_nodes[id] for id in corners]
        # Calculate the center of the corners
        new_x = sum(node[0] for node in new_corners) / len(new_corners)
        new_z = sum(node[2] for node in new_corners) / len(new_corners)
        # Update
        new_centers[i][0] = new_x
        new_centers[i][2] = new_z

    return new_centers


def write(points: list, original_grid, filename: str):
    # Create a deep copy of the original grid to preserve all data
    new_grid = vtk.vtkUnstructuredGrid()  # type: ignore
    new_grid.DeepCopy(original_grid)
    # Only modify the points ad not the rest of the file
    new_points = vtk.vtkPoints() # type: ignore
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


def process(
    path_to_nodes: pathlib.Path,
    path_to_centers: pathlib.Path,
    filename_nodes: str,
    filename_centers: str,
) -> None:
    # Modify the nodes mesh
    nodes_points, original_grid = load(path_to_nodes)
    new_points = modify_nodes(nodes_points)
    write(new_points, original_grid, filename_nodes)
    # Modify the centers mesh
    centers_points, original_grid = load(path_to_centers)
    new_centers = modify_centers(centers_points, nodes_points, new_points)
    write(new_centers, original_grid, filename_centers)
    return None


def plot(
    path_to_nodes: pathlib.Path, path_to_centers: pathlib.Path, filename: str
) -> None:
    # Plot mesh as a grid
    nodes, _ = load(path_to_nodes)
    centers, _ = load(path_to_centers)
    nodes = np.array(nodes)
    centers = np.array(centers)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.scatter(
        nodes[:, 0], nodes[:, 2], c="blue", s=50, label="Nodes", marker="o"
    )  # x, z coordinates
    ax.scatter(
        centers[:, 0], centers[:, 2], c="red", s=50, label="Centers", marker="x"
    )  # x, z coordinates

    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Z", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    # ax.legend()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    process(
        PATH_TO_MESHES / "fluid_nodes_fastest_256.vtu",
        PATH_TO_MESHES / "fluid_centers_fastest_256.vtu",
        "fluid_nodes_fastest_256_stretched.vtu",
        "fluid_centers_fastest_256_stretched.vtu",
    )
    plot(
        PATH_TO_MESHES / "fluid_nodes_fastest_256.vtu",
        PATH_TO_MESHES / "fluid_centers_fastest_256.vtu",
        "original_mesh",
    )
    plot(
        PATH_TO_STRETCHED / "fluid_nodes_fastest_256_stretched.vtu",
        PATH_TO_STRETCHED / "fluid_centers_fastest_256_stretched.vtu",
        "new_mesh",
    )
    # plot(PATH_TO_MESHES / "fluid_centers_fastest_256.vtu", PATH_TO_STRETCHED / "fluid_centers_fastest_256_stretched.vtu", "centers")
    # plot(PATH_TO_MESHES / "fluid_nodes_fastest_256.vtu", PATH_TO_STRETCHED / "fluid_nodes_fastest_256_stretched.vtu", "nodes")
