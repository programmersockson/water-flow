import numpy as np


def num_blues(simplex: np.ndarray):
    num = 0
    for p in simplex:
        if p >= 42:
            num += 1
    return num


def find_neighbors(ind: int, simplices: list):
    neighbors = []
    for simp in simplices:
        if ind in simp:
            simp_neighbors = list(simp).copy()
            simp_neighbors.remove(ind)
            neighbors.extend(simp_neighbors)
    return set(neighbors)


def find_areas(all_points: np.ndarray, simplices: list):
    areas = []
    for simplex in simplices:
        x0, y0 = all_points[simplex[0]]
        x1, y1 = all_points[simplex[1]]
        x2, y2 = all_points[simplex[2]]
        area = x1*y2 - x2*y1 - x0*y2 + x2*y0 + x0*y1 - x1*y0
        areas.append(area)
    return np.array(areas)


def find_velocities(epures: list, points: np.ndarray):
    velocities = []
    for i in range(20):
        velocities.append(np.array([0, 0]))
    for i in range(20, 22):
        a, b, c = epures[0][2:]
        x0, y0 = epures[0][0], points[i][1]
        x = -(a * y0 ** 2 + b * y0 + c)
        velocities.append(np.array([x - x0, 0]))
    for i in range(22, len(points)):
        velocities.append(np.array([0, 0]))
    return velocities


def step(points: np.ndarray, velocities: np.ndarray, simplices: list, dt: float = 0.5, epochs=1000):
    S = np.array(find_areas(points, simplices))
    nextS = np.array(find_areas(points + velocities, simplices))
    dS = nextS - S
    dVi = [[] for _ in range(len(points))]  # for points
    # dVi = np.zeros((len(velocities), 2))
    dVs = np.sign(dS) * np.sqrt(np.abs(dS))  # for simplex
    Loss = [sum(abs(dS))]
    for i in range(0, epochs):

        for i in range(len(simplices)):
            blues = num_blues(simplices[i])
            if blues:
                scaling_factor = dVs[i] / blues
            else:
                scaling_factor = 0
            a_ind = simplices[i][0]
            b_ind = simplices[i][1]
            c_ind = simplices[i][2]
            a = points[a_ind]
            b = points[b_ind]
            c = points[c_ind]

            center = np.mean([a, b, c], axis=0)

            if a_ind >= 42 or a_ind in (20, 21):
                dVi[a_ind].append((center - a) * scaling_factor * dt)
                # dVi[a_ind] += (center - a) * scaling_factor
            if b_ind >= 42 or b_ind in (20, 21):
                dVi[b_ind].append((center - b) * scaling_factor * dt)
                # dVi[b_ind] += (center - b) * scaling_factor
            if c_ind >= 42 or c_ind in (20, 21):
                dVi[c_ind].append((center - c) * scaling_factor * dt)
                # dVi[c_ind] += (center - c) * scaling_factor

        dV = np.array([np.mean(dVi[i], axis=0) if i >= 42 else np.array([0, 0]) for i in range(len(dVi))])
        nextS = np.array(find_areas(points + velocities + dV, simplices))
        dS = nextS - S
        dVs = np.sign(dS) * np.sqrt(np.abs(dS))
        Loss.append(sum(abs(dS)))
        velocities += dV
        dVi = [[] for _ in range(len(points))]

    return velocities, Loss
