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


print(11 - (11 // 2 + 1))


def load(path_to_mesh: pathlib.Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName("input.vtu")
    reader.Update()
    ugrid = reader.GetOutput()
    points = ugrid.GetPoints()
    extract = []  # x,y,z,id
    for i in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(i)
        extract.append([x, y, z, i])
    return extract

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
    
def write(points: list, filename: str):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName("output.vtu")
    writer.SetInputData(points)
    writer.Write()
