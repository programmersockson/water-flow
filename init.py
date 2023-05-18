import flow
from math import sqrt, pi, ceil, floor, sin, cos, atan2
from random import random
import numpy as np


def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def centroid(a, b, c):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    x3 = c[0]
    y3 = c[1]
    ab = euclidean_distance(a, b)
    bc = euclidean_distance(b, c)
    ca = euclidean_distance(c, a)
    p = ab + bc + ca
    return np.array([(ab * x1 + bc * x2 + ca * x3) / p,
                     (ab * y1 + bc * y2 + ca * y3) / p])


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random):
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * #
    # https://github.com/emulbreh/bridson/blob/master/bridson/__init__.py #
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * #
    tau = 2 * pi
    cellsize = r / sqrt(2)

    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = np.array([px, py])
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    return [p for p in grid if p is not None]


def border_check(sites_bad: list, border: list) -> list:
    sites_better = []
    for site in sites_bad:
        good = True
        for limiter in border:
            good = (np.linalg.norm(site - limiter)) > 0.15
            if not good:
                break
        if good:
            sites_better.append(site)
    return sites_better


def get_sites_cosine(sites_better: list) -> list:
    sites = []
    for site in sites_better:
        if site[1] > cos(site[0]) + 1:
            sites.append(site)
    return sites


def check_simplices(simplices) -> list:
    good_simplices = []
    for simp in simplices:
        # check if all 3 points of the simplex belong to the lower boundary, if not, store in good_simplices
        good = True

        # sometimes bad with [0, 2]. [0, 2] is [0] and can connect only to [1] point, [20], or [46:]
        if 0 in simp:
            inner1 = 1 in simp and 20 in simp
            inner2 = 1 in simp and max(simp) >= 44
            inner3 = 20 in simp and max(simp) >= 44

            if not (inner1 or inner2 or inner3):
                good = False

        if simp[0] < 20:
            if simp[1] < 20:
                if simp[2] < 20:
                    good = False
        if good:
            good_simplices.append(simp)
    return good_simplices


def counterclock_simplices(simplices: list, points: np.ndarray) -> list:
    new_simplices = []
    for simplice in simplices:
        a = points[simplice[0]]
        b = points[simplice[1]]
        c = points[simplice[2]]
        o = np.mean([a, b, c], axis=0)
        angles = [atan2(y - o[1], x - o[0]) for x, y in [a, b, c]]
        counterclockwise_indices = sorted(range(3), key=lambda i: angles[i])
        counterclockwise_simplex = np.array([[simplice[0], simplice[1], simplice[2]][i] for i in counterclockwise_indices])
        new_simplices.append(counterclockwise_simplex)
    return new_simplices


def areas_fix(all_points: np.ndarray, simplices: list, perfect_area: float, repeats=100):
    # if not bordering, move all three points to or from center of triangle, if bordering, move only not bordering ones
    for _ in range(repeats):
        # move the most deviated one, then recalculate the areas, repeat
        Ss = flow.find_areas(all_points, simplices)
        dSs = Ss - perfect_area
        # if max(dSs) < 0.01:
        #     break
        worst_ind = np.argmax(abs(dSs))
        worst = simplices[worst_ind]
        a_ind = worst[0]
        b_ind = worst[1]
        c_ind = worst[2]
        a = all_points[a_ind]
        b = all_points[b_ind]
        c = all_points[c_ind]
        center = np.mean([a, b, c], axis=0)
        if dSs[worst_ind] < 0:
            scaling_factor = -sqrt(perfect_area / Ss[worst_ind]) / 5
        else:
            scaling_factor = sqrt(perfect_area / Ss[worst_ind]) / 5
        if a_ind < 44:
            all_points[a_ind] = a
        else:
            all_points[a_ind] = a + (center - a) * scaling_factor

        if b_ind < 44:
            all_points[b_ind] = b
        else:
            all_points[b_ind] = b + (center - b) * scaling_factor

        if c_ind < 44:
            all_points[c_ind] = c
        else:
            all_points[c_ind] = c + (center - c) * scaling_factor

    return all_points


def find_coefficients(vertex: list, point1: list, point2: list):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = vertex

    m = np.array([[y1 ** 2., y1, 1.],
                  [y2 ** 2., y2, 1.],
                  [y3 ** 2., y3, 1.]])
    v = np.array([-x1, -x2, -x3])

    return np.linalg.solve(m, v)


def make_epures(top_border: list, bottom_border: list, qbp: int) -> list:
    epures = []
    for i in range(qbp):
        a, b, c = find_coefficients(
            [top_border[i][0] + 2 * np.pi / (qbp - 1), (top_border[i][1] + bottom_border[i][1]) / 2],
            bottom_border[i], top_border[i])
        x = top_border[i][0]
        if i != qbp - 1:
            x_next = top_border[i + 1][0]
        else:
            x_next = top_border[qbp - 1][0] + 2 * np.pi / (qbp - 1)
        epures.append([x, x_next, a, b, c])

    return epures