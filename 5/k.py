# 9. Write a program to do the following: You have given a collection of 8
# points. P1=[0.1,0.6] P2=[0.15,0.71] P3=[0.08,0.9] P4=[0.16, 0.85]
# P5=[0.2,0.3] P6=[0.25,0.5] P7=[0.24,0.1] P8=[0.3,0.2]. Perform the k-mean
# clustering with initial centroids as m1=P1 =Cluster#1=C1 and
# m2=P8=cluster#2=C2. Answer the following 1] Which cluster does P6
# belong to? 2] What is the population of a cluster around m2? 3] What is
# the updated value of m1 and m2?


#9

import numpy as np

# Given points
points = {
    'P1': np.array([0.1, 0.6]),
    'P2': np.array([0.15, 0.71]),
    'P3': np.array([0.08, 0.9]),
    'P4': np.array([0.16, 0.85]),
    'P5': np.array([0.2, 0.3]),
    'P6': np.array([0.25, 0.5]),
    'P7': np.array([0.24, 0.1]),
    'P8': np.array([0.3, 0.2])
}

# Initial centroids
m1 = np.array([0.1, 0.6])  # Centroid for Cluster #1
m2 = np.array([0.3, 0.2])  # Centroid for Cluster #2

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to assign points to the closest centroid
def assign_to_cluster(points, m1, m2):
    cluster_1 = []
    cluster_2 = []
    for key, point in points.items():
        distance_to_m1 = euclidean_distance(point, m1)
        distance_to_m2 = euclidean_distance(point, m2)

        if distance_to_m1 < distance_to_m2:
            cluster_1.append(key)
        else:
            cluster_2.append(key)

    return cluster_1, cluster_2

# Function to update centroids
def update_centroid(points, cluster):
    cluster_points = [points[key] for key in cluster]
    return np.mean(cluster_points, axis=0)

# First assignment of points to clusters
cluster_1, cluster_2 = assign_to_cluster(points, m1, m2)

# Update centroids
new_m1 = update_centroid(points, cluster_1)
new_m2 = update_centroid(points, cluster_2)

# Output
print(f"Initial clusters:")
print(f"Cluster 1: {cluster_1} | Centroid: {m1}")
print(f"Cluster 2: {cluster_2} | Centroid: {m2}")

print(f"\nUpdated clusters after one iteration:")
print(f"Cluster 1: {cluster_1} | New Centroid: {new_m1}")
print(f"Cluster 2: {cluster_2} | New Centroid: {new_m2}")

# Answering the specific questions

# 1. Which cluster does P6 belong to?
distance_to_m1 = euclidean_distance(points['P6'], m1)
distance_to_m2 = euclidean_distance(points['P6'], m2)
if distance_to_m1 < distance_to_m2:
    print(f"\nP6 belongs to Cluster 1")
else:
    print(f"\nP6 belongs to Cluster 2")

# 2. What is the population of a cluster around m2?
print(f"\nPopulation around m2 (Cluster 2): {len(cluster_2)}")

# 3. What is the updated value of m1 and m2?
print(f"\nUpdated m1: {new_m1}")
print(f"Updated m2: {new_m2}")



























# 10. Write a program to do the following: You have given a collection of 8
# points. P1=[2, 10] P2=[2, 5] P3=[8, 4] P4=[5, 8] P5=[7,5] P6=[6, 4] P7=[1, 2]
# P8=[4, 9]. Perform the k-mean clustering with initial centroids as m1=P1
# =Cluster#1=C1 and m2=P4=cluster#2=C2, m3=P7 =Cluster#3=C3. Answer
# the following 1] Which cluster does P6 belong to? 2] What is the
# population of a cluster around m3? 3] What is the updated value of m1,
# m2, m3?



#10

import numpy as np

# Given points
points = {
    'P1': np.array([2, 10]),
    'P2': np.array([2, 5]),
    'P3': np.array([8, 4]),
    'P4': np.array([5, 8]),
    'P5': np.array([7, 5]),
    'P6': np.array([6, 4]),
    'P7': np.array([1, 2]),
    'P8': np.array([4, 9])
}

# Initial centroids
m1 = np.array([2, 10])  # Centroid for Cluster #1
m2 = np.array([5, 8])   # Centroid for Cluster #2
m3 = np.array([1, 2])   # Centroid for Cluster #3

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to assign points to the closest centroid
def assign_to_cluster(points, m1, m2, m3):
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    for key, point in points.items():
        distance_to_m1 = euclidean_distance(point, m1)
        distance_to_m2 = euclidean_distance(point, m2)
        distance_to_m3 = euclidean_distance(point, m3)

        # Assign to the closest centroid
        if distance_to_m1 < distance_to_m2 and distance_to_m1 < distance_to_m3:
            cluster_1.append(key)
        elif distance_to_m2 < distance_to_m1 and distance_to_m2 < distance_to_m3:
            cluster_2.append(key)
        else:
            cluster_3.append(key)

    return cluster_1, cluster_2, cluster_3

# Function to update centroids
def update_centroid(points, cluster):
    cluster_points = [points[key] for key in cluster]
    return np.mean(cluster_points, axis=0)

# First assignment of points to clusters
cluster_1, cluster_2, cluster_3 = assign_to_cluster(points, m1, m2, m3)

# Update centroids
new_m1 = update_centroid(points, cluster_1)
new_m2 = update_centroid(points, cluster_2)
new_m3 = update_centroid(points, cluster_3)

# Output the clusters and centroids
print(f"Initial clusters:")
print(f"Cluster 1: {cluster_1} | Centroid: {m1}")
print(f"Cluster 2: {cluster_2} | Centroid: {m2}")
print(f"Cluster 3: {cluster_3} | Centroid: {m3}")

print(f"\nUpdated clusters after one iteration:")
print(f"Cluster 1: {cluster_1} | New Centroid: {new_m1}")
print(f"Cluster 2: {cluster_2} | New Centroid: {new_m2}")
print(f"Cluster 3: {cluster_3} | New Centroid: {new_m3}")

# Answering the specific questions

# 1. Which cluster does P6 belong to?
distance_to_m1 = euclidean_distance(points['P6'], m1)
distance_to_m2 = euclidean_distance(points['P6'], m2)
distance_to_m3 = euclidean_distance(points['P6'], m3)
if distance_to_m1 < distance_to_m2 and distance_to_m1 < distance_to_m3:
    print(f"\nP6 belongs to Cluster 1")
elif distance_to_m2 < distance_to_m1 and distance_to_m2 < distance_to_m3:
    print(f"\nP6 belongs to Cluster 2")
else:
    print(f"\nP6 belongs to Cluster 3")

# 2. What is the population of a cluster around m3?
print(f"\nPopulation around m3 (Cluster 3): {len(cluster_3)}")

# 3. What is the updated value of m1, m2, m3?
print(f"\nUpdated m1: {new_m1}")
print(f"Updated m2: {new_m2}")
print(f"Updated m3: {new_m3}")



























# 20. Write a program to cluster a set of points using K-means for IRIS
# dataset. Consider, K=3, clusters. Consider Euclidean distance as the
# distance measure. Randomly initialize a cluster mean as one of the data
# points. Iterate at least for 10 iterations. After iterations are over, print the
# final cluster means for each of the clusters.


#20

import pandas as pd
import numpy as np

# Load the Iris dataset
url = '/content/IRIS.csv'
iris_data = pd.read_csv(url)

# Use only the numerical columns for clustering
X = iris_data.iloc[:, :-1].values  # Exclude the species column

# Set parameters
K = 3  # Number of clusters
max_iterations = 10  # Number of iterations

# Randomly initialize cluster centroids as one of the data points
np.random.seed(42)  # For reproducibility
centroids = X[np.random.choice(X.shape[0], K, replace=False)]

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# K-means clustering
for iteration in range(max_iterations):
    # Step 1: Assign each point to the nearest cluster
    clusters = [[] for _ in range(K)]
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(point)

    # Step 2: Update centroids to the mean of the assigned points
    new_centroids = []
    for cluster_points in clusters:
        new_centroids.append(np.mean(cluster_points, axis=0) if cluster_points else np.zeros(X.shape[1]))
    new_centroids = np.array(new_centroids)

    # Check for convergence (optional)
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# Print final cluster means
print("Final cluster means after {} iterations:".format(max_iterations))
for idx, centroid in enumerate(centroids, start=1):
    print(f"Cluster {idx}: {centroid}")




























# 21. Write a program to cluster a set of points using K-means for IRIS
# dataset. Consider, K=4, clusters. Consider Euclidean distance as the
# distance measure. Randomly initialize a cluster mean as one of the data
# points. Iterate at least for 10 iterations. After iterations are over, print the
# final cluster means for each of the clusters.




#21

import pandas as pd
import numpy as np

# Load the Iris dataset
url = '/content/IRIS.csv'
iris_data = pd.read_csv(url)

# Use only the numerical columns for clustering
X = iris_data.iloc[:, :-1].values  # Exclude the species column

# Set parameters
K = 4  # Number of clusters
max_iterations = 10  # Number of iterations

# Randomly initialize cluster centroids as one of the data points
np.random.seed(42)  # For reproducibility
centroids = X[np.random.choice(X.shape[0], K, replace=False)]

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# K-means clustering
for iteration in range(max_iterations):
    # Step 1: Assign each point to the nearest cluster
    clusters = [[] for _ in range(K)]
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(point)

    # Step 2: Update centroids to the mean of the assigned points
    new_centroids = []
    for cluster_points in clusters:
        new_centroids.append(np.mean(cluster_points, axis=0) if cluster_points else np.zeros(X.shape[1]))
    new_centroids = np.array(new_centroids)

    # Check for convergence (optional)
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# Print final cluster means
print("Final cluster means after {} iterations:".format(max_iterations))
for idx, centroid in enumerate(centroids, start=1):
    print(f"Cluster {idx}: {centroid}")






























# ambarish graph
#9
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define the points as an array
points = np.array([
    [0.1, 0.6],  # P1
    [0.15, 0.71],  # P2
    [0.08, 0.9],  # P3
    [0.16, 0.85],  # P4
    [0.2, 0.3],  # P5
    [0.25, 0.5],  # P6
    [0.24, 0.1],  # P7
    [0.3, 0.2]   # P8
])

# Initial centroids as m1=P1 and m2=P8
initial_centroids = np.array([[0.1, 0.6], [0.3, 0.2]])

# Perform K-means clustering with 2 clusters and given initial centroids
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1)
kmeans.fit(points)

# Get cluster assignments
labels = kmeans.labels_

# Get updated centroids
centroids = kmeans.cluster_centers_

# Plot the points and clusters
plt.figure(figsize=(8, 6))

# Scatter plot for points with different colors for each cluster
colors = ['red' if label == 0 else 'blue' for label in labels]
plt.scatter(points[:, 0], points[:, 1], c=colors, s=100, label='Points')

# Mark the initial centroids with 'x'
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], marker='x', s=200, c='green', label='Initial Centroids')

# Mark the updated centroids with a larger size and different marker
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='purple', label='Updated Centroids')

# Label the points
for i, point in enumerate(points):
    plt.text(point[0] + 0.02, point[1], f'P{i+1}', fontsize=12)

# Set plot labels and legend
plt.title('K-means Clustering of Points with Initial Centroids')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print results
print(f"Updated centroids: {centroids}")
print(f"P6 belongs to cluster: {'C1' if labels[5] == 0 else 'C2'}")
print(f"Population of cluster around m2 (C2): {sum(labels == 1)}")






#10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define the points as an array
points = np.array([
    [2, 10],  # P1
    [2, 5],   # P2
    [8, 4],   # P3
    [5, 8],   # P4
    [7, 5],   # P5
    [6, 4],   # P6
    [1, 2],   # P7
    [4, 9]    # P8
])

# Initial centroids as m1=P1, m2=P4, and m3=P7
initial_centroids = np.array([
    [2, 10],  # m1 = P1
    [5, 8],   # m2 = P4
    [1, 2]    # m3 = P7
])

# Perform K-means clustering with 3 clusters and given initial centroids
kmeans = KMeans(n_clusters=3, init=initial_centroids, n_init=1)
kmeans.fit(points)

# Get cluster assignments
labels = kmeans.labels_

# Get updated centroids
centroids = kmeans.cluster_centers_

# Plot the points and clusters
plt.figure(figsize=(8, 6))

# Scatter plot for points with different colors for each cluster
colors = ['red' if label == 0 else 'blue' if label == 1 else 'green' for label in labels]
plt.scatter(points[:, 0], points[:, 1], c=colors, s=100, label='Points')

# Mark the initial centroids with 'x'
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], marker='x', s=200, c='black', label='Initial Centroids')

# Mark the updated centroids with a larger size and different marker
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='purple', label='Updated Centroids')

# Label the points
for i, point in enumerate(points):
    plt.text(point[0] + 0.2, point[1], f'P{i+1}', fontsize=12)

# Set plot labels and legend
plt.title('K-means Clustering of Points with Initial Centroids')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print results
print(f"Updated centroids: {centroids}")
print(f"P6 belongs to cluster: {'C1' if labels[5] == 0 else 'C2' if labels[5] == 1 else 'C3'}")
print(f"Population of cluster around m3 (C3): {sum(labels == 2)}")










# manglesh
import numpy as np

# Define the points
points = np.array([
    [0.1, 0.6],  # P1
    [0.15, 0.71],  # P2
    [0.08, 0.9],  # P3
    [0.16, 0.85],  # P4
    [0.2, 0.3],  # P5
    [0.25, 0.5],  # P6
    [0.24, 0.1],  # P7
    [0.3, 0.2]   # P8
])

# Initial centroids
m1 = np.array([0.1, 0.6])  # P1
m2 = np.array([0.3, 0.2])  # P8
# Function to calculate Euclidean distance
def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))
# Perform one iteration of k-means clustering
def k_means_iteration(points, m1, m2):
    clusters = {1: [], 2: []}

    # Assign points to the nearest centroid
    for point in points:
        distance_to_m1 = euclidean_distance(point, m1)
        distance_to_m2 = euclidean_distance(point, m2)
        if distance_to_m1 < distance_to_m2:
            clusters[1].append(point)
        else:
            clusters[2].append(point)

    # Update centroids
    new_m1 = np.mean(clusters[1], axis=0)
    new_m2 = np.mean(clusters[2], axis=0)

    return clusters, new_m1, new_m2
# Perform the iteration
clusters, updated_m1, updated_m2 = k_means_iteration(points, m1, m2)
# Determine which cluster P6 belongs to
p6_cluster = 1 if euclidean_distance(points[5], updated_m1) < euclidean_distance(points[5], updated_m2) else 2
# Output results
print(f"P6 belongs to Cluster #{p6_cluster}")
print(f"Population of Cluster #2 around m2: {len(clusters[2])}")
print(f"Updated m1: {updated_m1}")
print(f"Updated m2: {updated_m2}")