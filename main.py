import init
import flow
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


qbp = 20  # quantity of border points

# points initialisation
top_border = [np.array([x * 2 * np.pi / (qbp - 1), 3]) for x in range(qbp)]
bottom_border_cos = [np.array([x / (qbp - 1) * 2 * np.pi, np.cos(x / np.pi) + 1]) for x in range(qbp)]
left_in = [np.array([0, y / 3 + 2]) for y in range(1, 3)]
right_out = [np.array([2 * np.pi, y / 3 + 2]) for y in range(1, 3)]
border = bottom_border_cos + left_in + top_border + right_out
sites_bad = init.poisson_disc_samples(2 * np.pi, 3, 0.3, 5)
sites_better = init.border_check(sites_bad, border)
sites = init.get_sites_cosine(sites_better)

# triangulation initialisation
all_points = np.array(border + sites)
triangulation = Delaunay(all_points)

# finding perfect area for this case
perfect_area_cos = (3 * 2 * np.pi - 2 * np.pi) / (len(triangulation.simplices))

# picking only good simplices
simplices = init.check_simplices(triangulation.simplices)
simplices = init.counterclock_simplices(simplices, all_points)


all_points = init.areas_fix(all_points, simplices, perfect_area_cos, repeats=1000)


# initialising velocities for each point
epures = init.make_epures(top_border, bottom_border_cos, qbp)
velocities0 = flow.find_velocities(epures, all_points)


# step
velocities, Loss = flow.step(all_points, velocities0, simplices)
velocitiesP = np.array([v / np.linalg.norm(v) if list(v) != [0, 0] else np.array([0, 0]) for v in velocities])  # to unit vectors
velocities = np.array([v / np.linalg.norm(v) for v in velocities])  # to unit vectors


# plot
fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2)
plt.subplot(1, 2, 1)
plt.gca().set_aspect('equal')

# triangulation
plt.triplot(np.array(all_points).T[0], np.array(all_points).T[1], simplices)
# velocities
plt.quiver(np.array(all_points).T[0], np.array(all_points).T[1], np.array(velocities).T[0], np.array(velocities).T[1], color='blue')

# sites
plt.scatter(np.array(all_points[44:]).T[0], np.array(all_points[44:]).T[1], color='blue')
plt.scatter(np.array(all_points[42:44]).T[0], np.array(all_points[42:44]).T[1], color='blue')
# static points
plt.scatter(np.array(all_points[:42]).T[0], np.array(all_points[:42]).T[1], color='red')


plt.title('Trained state', size=20)

plt.subplot(1, 2, 2)
# plt.gca().set_aspect('equal')
plt.plot(range(len(Loss)), Loss)
plt.title('Loss', size=20)

plt.show()


# print('all points (42 first are red)')
# print([list(p) for p in all_points])
# print()
# print('simplices')
# print([list(p) for p in simplices])
# print()
# print('velocities0')
# print([list(p) for p in velocities0])
# print()
# print('velocities')
# print([list(p) for p in velocitiesP])

