import flow
from math import sqrt, pi, ceil, floor, sin, cos, isclose
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
    return [(ab * x1 + bc * x2 + ca * x3) / p,
            (ab * y1 + bc * y2 + ca * y3) / p]


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random):
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
    # https://github.com/emulbreh/bridson/blob/master/bridson/__init__.py
    # - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
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
            p = [px, py]
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
            good = (np.linalg.norm(np.array(site) - np.array(limiter))) > 0.25
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
            inner2 = 1 in simp and max(simp) >= 46
            inner3 = 20 in simp and max(simp) >= 46

            if not (inner1 or inner2 or inner3):
                good = False

        if simp[0] < 20:
            if simp[1] < 20:
                if simp[2] < 20:
                    good = False
        if good:
            good_simplices.append(simp)
    return good_simplices


def areas_fix(all_points: list, simplices: list, perfect_area: float, repeats=1):
    # if not bordering, move all three points to or from center of triangle, if bordering, move only not bordering points
    for _ in range(repeats):
        # move the most deviated one, then recalculate the areas, repeat
        Ss = np.array(flow.find_areas(all_points, simplices))
        dSs = np.abs(Ss - perfect_area)
        # if max(dSs) < 0.01:
        #     break
        worst_ind = np.argmax(dSs)
        worst = simplices[worst_ind]
        a_ind = worst[0]
        b_ind = worst[1]
        c_ind = worst[2]
        a = np.array(all_points[a_ind])
        b = np.array(all_points[b_ind])
        c = np.array(all_points[c_ind])
        # center = np.mean([a, b, c], axis=0)
        center = centroid(a, b, c)
        scaling_factor = sqrt(perfect_area / flow.find_area([a, b, c])) / 5
        if a_ind < 46:
            all_points[a_ind] = list(a)
        else:
            all_points[a_ind] = list(a + (center - a) * scaling_factor)

        if b_ind < 46:
            all_points[b_ind] = list(b)
        else:
            all_points[b_ind] = list(b + (center - b) * scaling_factor)

        if c_ind < 46:
            all_points[c_ind] = list(c)
        else:
            all_points[c_ind] = list(c + (center - c) * scaling_factor)

    return all_points
