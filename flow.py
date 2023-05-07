import numpy as np


def find_areas(all_points: list, simplices: list):
    areas = []
    for simplex in simplices:
        a = all_points[simplex[0]]
        b = all_points[simplex[1]]
        c = all_points[simplex[2]]
        area = find_area([a, b, c])
        areas.append(area)
    return np.array(areas)


def find_area(points: list) -> float:
    a = np.array(points[0])
    b = np.array(points[1])
    c = np.array(points[2])
    area = 0.5 * np.abs(np.cross(b - a, c - a))
    return area


def find_velocities(epures: list, points: list):
    velocities = []
    for i in range(20):
        velocities.append(np.array([0, 0]))
    for i in range(20, 22):
        a, b, c = epures[0][2:]
        x0, y0 = epures[0][0], points[i][1]
        x = -(a * y0 ** 2 + b * y0 + c)
        velocities.append(np.array([x - x0, 0]))
    for i in range(22, 42):
        velocities.append(np.array([0, 0]))
    for i in range(42, 44):
        a, b, c = epures[-1][2:]
        x0, y0 = epures[-1][0], points[i][1]
        x = -(a * y0 ** 2 + b * y0 + c)
        velocities.append(np.array([x - x0, 0]))
    for i in range(44, len(points)):
        for epure in epures:
            if epure[0] <= points[i][0] < epure[1]:
                a, b, c = epure[2:]
                x0, y0 = epure[0], points[i][1]
                x = -(a * y0 ** 2 + b * y0 + c)
                velocities.append(np.array([x - x0, 0]))
    return velocities


def step():
    pass
