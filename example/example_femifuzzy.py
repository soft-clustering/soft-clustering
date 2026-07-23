from os import path
import sys
import numpy as np

if __name__ == "__main__":
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import FeMIFuzzy

    # Simulate 2 clients with incomplete data
    np.random.seed(0)
    clients = [np.random.rand(20, 5), np.random.rand(20, 5)]
    features = [["f1", "f2", "f3", "f4", "f5"], ["f1", "f2", "f3", "f4", "f5"]]

    model = FeMIFuzzy(random_state=0, max_iter=10)
    centroids = model.fit_predict(clients, features)

    print("Number of centroids:", len(centroids))
    print("Centroid shapes:", [c.shape for c in centroids])
