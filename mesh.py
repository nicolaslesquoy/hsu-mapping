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

def modify_nodes(nodes: list) -> list:
    # Create a deep copy of the original nodes to preserve all data
    new_nodes = [list(point) for point in nodes]  # Deep copy
    # Stretching the mesh (nodes only)
    bottom = filter_points(nodes, coordinate="z", target=min(p[2] for p in nodes))
    left = filter_points(nodes, coordinate="x", target=min(p[0] for p in nodes))
    middle_index = len(bottom) // 2
    L = (max(p[0] for p in nodes) - min(p[0] for p in nodes)) / 2
    s = 1.2
    dx = L * (1 - s) / (1 - s**(middle_index - 1))
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
            new_z = old_z + dx * s**(j - 1)
            new_nodes[column[j][3]][2] = new_z
            old_z = new_z
        # Reverse column
        column.reverse()
        # Stretch upper half
        zn = column[0][2]
        old_z = zn
        for j in range(1, middle_index):
            new_z = old_z - dx * s**(j - 1)
            new_nodes[column[j][3]][2] = new_z
            old_z = new_z
    # Stretching horizontally (x coordinate)
    for i in range(len(left)):
        row_pos = left[i][2]
        row_indices = [
            point[3] for point in nodes if abs(point[2] - row_pos) < 1e-6
        ]
        row = [nodes[idx] for idx in row_indices]
        row.sort(key=lambda p: p[0])
        # Stretch left half
        x0 = row[0][0]
        old_x = x0
        for j in range(1, middle_index):
            new_x = old_x + dx * s**(j - 1)
            new_nodes[row[j][3]][0] = new_x
            old_x = new_x
        # Reverse row
        row.reverse()
        # Stretch right half
        xn = row[0][0]
        old_x = xn
        for j in range(1, middle_index):
            new_x = old_x - dx * s**(j - 1)
            new_nodes[row[j][3]][0] = new_x
            old_x = new_x
    return new_nodes

def search_point(points: list, point: list) -> int:
    for i, p in enumerate(points):
        if all(abs(p[j] - point[j]) < 1e-6 for j in range(3)):
            return p[3]  # Return the index of the point
    return -1

def modify_centers(centers: list, old_nodes: list, new_nodes: list) -> list:
    new_centers = [list(point) for point in centers]  # Deep copy
    # Get bottom points for vertical columns
    bottom_old = filter_points(old_nodes, coordinate="z", target=min(p[2] for p in old_nodes))
    bottom_new = filter_points(new_nodes, coordinate="z", target=min(p[2] for p in new_nodes))
    # Get left points for horizontal rows
    left_old = filter_points(old_nodes, coordinate="x", target=min(p[0] for p in old_nodes))
    left_new = filter_points(new_nodes, coordinate="x", target=min(p[0] for p in new_nodes))
    
    for i in range(len(bottom_old)-1):
        # Create the columns
        column_pos = bottom_old[i][0]
        next_column_pos = bottom_old[i+1][0]
        # Get new x positions from the stretched mesh
        new_column_pos = new_nodes[bottom_new[i][3]][0]
        new_next_column_pos = new_nodes[bottom_new[i+1][3]][0]
        
        column_indices = [
            point[3] for point in old_nodes if abs(point[0] - column_pos) < 1e-6
        ]
        next_column_indices = [
            point[3] for point in old_nodes if abs(point[0] - next_column_pos) < 1e-6
        ]
        column = [old_nodes[idx] for idx in column_indices]
        next_column = [old_nodes[idx] for idx in next_column_indices]
        column.sort(key=lambda p: p[2])
        next_column.sort(key=lambda p: p[2])
        
        column_indices = [
            point[3] for point in new_nodes if abs(point[0] - new_column_pos) < 1e-6
        ]
        next_column_indices = [
            point[3] for point in new_nodes if abs(point[0] - new_next_column_pos) < 1e-6
        ]
        new_column = [new_nodes[idx] for idx in column_indices]
        new_next_column = [new_nodes[idx] for idx in next_column_indices]
        new_column.sort(key=lambda p: p[2])
        new_next_column.sort(key=lambda p: p[2])
        
        for j in range(len(column)-1):
            # Define the old center
            point_bottom_right = (next_column_pos, next_column[j][2])
            point_top_left = (column_pos, column[j+1][2])
            old_center = [
                (point_bottom_right[0] + point_top_left[0]) / 2,
                0,
                (point_bottom_right[1] + point_top_left[1]) / 2,
            ]  
            # Define the new center using stretched x coordinates
            new_point_bottom_right = (new_next_column_pos, new_next_column[j][2])
            new_point_top_left = (new_column_pos, new_column[j+1][2])
            new_center = [
                (new_point_bottom_right[0] + new_point_top_left[0]) / 2,
                0,
                (new_point_bottom_right[1] + new_point_top_left[1]) / 2,
            ]
            
            old_center_index = search_point(centers, old_center)
            if old_center_index != -1:
                new_centers[old_center_index][0] = new_center[0]
                new_centers[old_center_index][2] = new_center[2]
    print("number of centers modified:", len(new_centers))
    print("old centers:", len(centers))
    return new_centers
    


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


def process(path_to_nodes: pathlib.Path, path_to_centers: pathlib.Path, filename_nodes: str, filename_centers: str) -> None:
    # Modify the nodes mesh
    nodes_points, original_grid = load(path_to_nodes)
    new_points = modify_nodes(nodes_points)
    write(new_points, original_grid, filename_nodes)
    # Modify the centers mesh
    centers_points, original_grid = load(path_to_centers)
    new_centers = modify_centers(centers_points, nodes_points, new_points)
    write(new_centers, original_grid, filename_centers)
    return None

def plot(path_to_nodes: pathlib.Path, path_to_centers: pathlib.Path, filename: str) -> None:
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
    plt.savefig(
        f"{filename}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    

if __name__ == "__main__":
    # process(
    #     PATH_TO_MESHES / "fluid_nodes_fastest_32.vtu", "fluid_nodes_fastest_32_stretched.vtu"
    # )
    # plot(PATH_TO_STRETCHED / "fluid_nodes_fastest_32_stretched.vtu")
    process(
        PATH_TO_MESHES / "fluid_nodes_fastest_32.vtu", PATH_TO_MESHES / "fluid_centers_fastest_32.vtu", "fluid_nodes_fastest_32_stretched.vtu", "fluid_centers_fastest_32_stretched.vtu"
    )
    plot(PATH_TO_MESHES / "fluid_nodes_fastest_32.vtu", PATH_TO_MESHES / "fluid_centers_fastest_32.vtu", "original_mesh")
    plot(PATH_TO_STRETCHED / "fluid_nodes_fastest_32_stretched.vtu", PATH_TO_STRETCHED / "fluid_centers_fastest_32_stretched.vtu", "new_mesh")