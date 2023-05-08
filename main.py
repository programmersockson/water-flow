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
all_points = border + sites
triangulation = Delaunay(all_points)

# finding perfect area for this case
perfect_area_cos = (3 * 2 * np.pi - 2 * np.pi) / (len(triangulation.simplices))
print('Goal simplex area: ', perfect_area_cos)

# picking only good simplices
simplices = init.check_simplices(triangulation.simplices)

# fixing areas of every simplex to the point they are all the same
areas = flow.find_areas(all_points, simplices)
print('Standard deviation of simplex area before: ', np.std(areas))
print('Max simplex area before: ', np.max(areas))
print('Min simplex area before: ', np.min(areas))
all_points = init.areas_fix(all_points, simplices, perfect_area_cos, repeats=1000)

areas = flow.find_areas(all_points, simplices)
print('Standard deviation of simplex area after: ', np.std(areas))
print('Max simplex area after: ', np.max(areas))
print('Min simplex area after: ', np.min(areas))


# initialising velocities for each point
epures = init.make_epures(top_border, bottom_border_cos, qbp)
velocities = flow.find_velocities(epures, all_points)

# plot
fig, ax = plt.subplots(figsize=(20, 10))
# plt.subplot(1, 2, 1)
plt.gca().set_aspect('equal')

# triangulation
plt.triplot(np.array(all_points).T[0], np.array(all_points).T[1], simplices)
# velocities
plt.quiver(np.array(all_points).T[0], np.array(all_points).T[1], np.array(velocities).T[0], np.array(velocities).T[1], color='lightblue')
# sites
plt.scatter(np.array(all_points[44:]).T[0], np.array(all_points[44:]).T[1], color='blue')
plt.scatter(np.array(all_points[42:44]).T[0], np.array(all_points[42:44]).T[1], color='blue')
# static points
plt.scatter(np.array(all_points[:42]).T[0], np.array(all_points[:42]).T[1], color='red')


plt.title('Initialisation', size=20)

# plt.subplot(1, 2, 2)
# plt.gca().set_aspect('equal')
#
# plt.title('One shift', size=20)
fig.show()
