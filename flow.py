import numpy as np


def find_areas(all_points: list, simplices: list) -> list:
    areas = []
    for simplex in simplices:
        a = np.array(all_points[simplex[0]])
        b = np.array(all_points[simplex[1]])
        c = np.array(all_points[simplex[2]])
        area = find_area([a, b, c])
        areas.append(area)
    return areas


def find_area(points: list) -> float:
    a = np.array(points[0])
    b = np.array(points[1])
    c = np.array(points[2])
    area = 0.5 * np.abs(np.cross(b - a, c - a))
    return area

