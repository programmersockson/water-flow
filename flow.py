import numpy as np


def find_areas(all_points: list, simplices: list):
    areas = []
    for simplex in simplices:
        a = all_points[simplex[0]]
        b = all_points[simplex[1]]
        c = all_points[simplex[2]]
        area = 0.5 * np.abs(np.cross(b - a, c - a))
        areas.append(area)
    return np.array(areas)


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


def step(points: list, velocities: list, simplices: list):
    S = np.array(find_areas(points, simplices))
    next_points = np.array(points) + np.array(velocities)
    nextS = np.array(find_areas(next_points, simplices))
    dS = S - nextS
    while np.allclose(dS, [0] * len(simplices)):
        for i, simplex in enumerate(simplices):
            pass
    return next_points, simplices