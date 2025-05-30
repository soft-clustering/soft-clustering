import sys
from os import path
import random

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import PFCM

#Create simple sample data (3 small groups of 2D points)
def create_data():
    data = []

    # Group 1: around (1, 1)
    for _ in range(5):
        x = 1 + random.uniform(-0.1, 0.1)
        y = 1 + random.uniform(-0.1, 0.1)
        data.append([x, y])

    # Group 2: around (5, 5)
    for _ in range(5):
        x = 5 + random.uniform(-0.1, 0.1)
        y = 5 + random.uniform(-0.1, 0.1)
        data.append([x, y])

    # Group 3: around (9, 1)
    for _ in range(5):
        x = 9 + random.uniform(-0.1, 0.1)
        y = 1 + random.uniform(-0.1, 0.1)
        data.append([x, y])

    return data

#Run PFCM on the data and print results
def run_pfcm():
    data = create_data()
    model = PFCM(n_clusters=3, random_state=0)
    model.fit(data)

    print("Cluster centers:")
    for i, center in enumerate(model.cluster_centroids):
        print(f"Cluster {i + 1}: {center}")

    print("\nMemberships for first 5 points:")
    for i in range(5):
        memberships = [round(model.membership_matrix[j][i], 3) for j in range(model.n_clusters)]
        print(f"Point {i + 1}: {memberships}")

    print("\nTypicalities for first 5 points:")
    for i in range(5):
        typicalities = [round(model.typicality_matrix[j][i], 3) for j in range(model.n_clusters)]
        print(f"Point {i + 1}: {typicalities}")


if __name__ == "__main__":
    run_pfcm()