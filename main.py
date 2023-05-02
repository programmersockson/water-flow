import init
import flow
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

qbp = 20  # quantity of border points

top_border = [[x * 2 * np.pi / (qbp - 1), 3] for x in range(qbp)]
bottom_border_cos = [[x / (qbp - 1) * 2 * np.pi, np.cos(x / np.pi) + 1] for x in range(qbp)]
left_in = [[0, y / 4 + 2] for y in range(1, 4)]
right_out = [[2 * np.pi, y / 4 + 2] for y in range(1, 4)]
border = bottom_border_cos + left_in + top_border + right_out
x_border = [limiter[0] for limiter in border]
y_border = [limiter[1] for limiter in border]

sites_bad = init.poisson_disc_samples(2 * np.pi, 3, 0.3, 5)
sites_better = init.border_check(sites_bad, border)
sites = init.get_sites_cosine(sites_better)
x_sites = [site[0] for site in sites]
y_sites = [site[1] for site in sites]

all_points = border + sites
x_points = [p[0] for p in all_points]
y_points = [p[1] for p in all_points]

triangulation = Delaunay(all_points)

perfect_area_cos = (3 * 2 * np.pi - 2 * np.pi) / (len(triangulation.simplices))
print('Goal simplex area: ', perfect_area_cos)

good_simplices = init.check_simplices(triangulation.simplices)

areas = flow.find_areas(all_points, good_simplices)
print('Standard deviation of simplex area before: ', np.std(areas))

# fig, ax = plt.subplots(figsize=(10, 5))
fig, ax = plt.subplots(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.gca().set_aspect('equal')
plt.triplot(x_points, y_points, good_simplices)
plt.scatter(x_sites, y_sites, color='blue')
plt.scatter(x_border, y_border, color='red')
plt.title('Delaunay triangulation')


plt.subplot(1, 2, 2)
plt.gca().set_aspect('equal')
all_points = init.areas_fix(all_points, good_simplices, perfect_area_cos, repeats=500)
areas = flow.find_areas(all_points, good_simplices)
print('Standard deviation of simplex area after: ', np.std(areas))
x_points = [p[0] for p in all_points]
y_points = [p[1] for p in all_points]
plt.triplot(x_points, y_points, good_simplices)
x_sites = [site[0] for site in all_points[46:]]
y_sites = [site[1] for site in all_points[46:]]
plt.scatter(x_sites, y_sites, color='blue')
plt.scatter(x_border, y_border, color='red')
plt.title('Треугольники Петросяна')

fig.show()

