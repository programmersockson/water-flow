import numpy as np
import init


def find_areas(all_points: np.ndarray, simplices: list):
    areas = []
    for simplex in simplices:
        a = all_points[simplex[0]]
        b = all_points[simplex[1]]
        c = all_points[simplex[2]]
        area = 0.5 * np.abs(np.cross(b - a, c - a))
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


def step(points: np.ndarray, velocities: np.ndarray, simplices: list):
    S = np.array(find_areas(points, simplices))
    nextS = np.array(find_areas(points + velocities, simplices))
    dS = S - nextS
    # dV = np.zeros((len(points), 2))
    # while not np.allclose(dS, np.zeros(len(simplices))):
    # while np.sum(abs(dS)) > 1:
    dVs = [[] for _ in range(len(points))]
    Loss = [sum(abs(dS))]

    for epoch in range(1, 100):

        for i in range(len(simplices)):
            a_ind = simplices[i][0]
            b_ind = simplices[i][1]
            c_ind = simplices[i][2]
            a = points[a_ind]
            b = points[b_ind]
            c = points[c_ind]

            center = init.centroid(a, b, c)

            scale = -100 if dS[i] < 0 else 100
            scaling_factor = np.sqrt(S[i] / nextS[i]) / scale

            if a_ind >= 42:
                dVs[a_ind].append((center - a) * scaling_factor)

            if b_ind >= 42:
                dVs[b_ind].append((center - b) * scaling_factor)

            if c_ind >= 42:
                dVs[c_ind].append((center - c) * scaling_factor)

        dV = np.array([np.mean(dVs[i], axis=0) if i >= 42 else np.array([0, 0]) for i in range(len(dVs))])
        nextS = np.array(find_areas(points + velocities + dV, simplices))
        dS = S - nextS
        Loss.append(sum(abs(dS)))

        # if Loss[-1] < Loss[-2]:
        #     velocities += dV
        #     dVs = [[] for _ in range(len(points))]
        # else:
        #     break
        velocities += dV
        dVs = [[] for _ in range(len(points))]


    return velocities, Loss
