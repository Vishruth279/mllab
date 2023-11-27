import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [5.9, 3.2],
    [4.6, 2.9],
    [6.2, 2.8],
    [4.7, 3.2],
    [5.5, 4.2],
    [5.0, 3.0],
    [4.9, 3.1],
    [6.7, 3.1],
    [5.1, 3.8],
    [6.0, 3.0]
])

initial_centers = np.array([
    [6.2, 3.2],  # Red cluster
    [6.6, 3.7],  # Green cluster
    [6.5, 3.0]   # Blue cluster
])
k = 3
max_iterations = 100

cluster_centers = initial_centers.copy()
for iteration in range(max_iterations):
    distances = np.linalg.norm(X[:, np.newaxis, :] - cluster_centers, axis=2)
    labels = np.argmin(distances, axis=1)
    new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    if np.allclose(cluster_centers, new_centers, rtol=1e-4):
        break
    cluster_centers = new_centers

# Print the final cluster centers and assigned points
print("Final cluster centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i + 1} Center: {center}")
    print(f"Points in Cluster {i + 1}:")
    for j, point in enumerate(X[labels == i]):
        print(f" Point {j + 1}: {point}")
    print()

# Scatter plot
plt.scatter(X[:, 0], X[:, 1], c=labels, marker='o', edgecolors='k', label='Data Points')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
