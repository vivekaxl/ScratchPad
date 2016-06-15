import numpy as np
raw_points = np.array([[1, 3],[3, 3],[3, 1],[1, 1]])

# Axis scaling
A = np.array([[4, -1], [0, 1]])

print raw_points
print A

points = np.dot(raw_points, A)
import matplotlib.pyplot as plt
plt.scatter(points[:, 0], points[:, 1])
plt.show()