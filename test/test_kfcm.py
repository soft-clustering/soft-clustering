import sys
from os import path
import numpy as np

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import KFCM
    
test_data_list = [
    [1.2, 1.1], [1.6, 2.0], [1.0, 1.3], [1.4, 1.9], # Cluster 1
    [6.1, 6.3], [6.6, 7.0], [6.0, 6.2], [6.9, 6.4], # Cluster 2
    [9.5, 2.1], [10.1, 2.6], [9.9, 1.9], [9.0, 2.3]  # Cluster 3
]
# The KFCM class expects a NumPy array as input
X = np.array(test_data_list)
np.random.shuffle(X)
print(f"\nCreated a test dataset with {X.shape[0]} points and {X.shape[1]} features.")

# 2. Instantiate and run the KFCM algorithm from your library
print("Instantiating and fitting the KFCM model...")
kfcm_instance = KFCM(
    n_clusters=3,
    m=2.0,
    sigma=3.0, # This parameter often needs tuning
    epsilon=0.001,
    max_iter=100
)
    
predicted_labels = kfcm_instance.fit(X)

# 3. Display the results
print("\n--- KFCM Test Results ---")
    
# Print the final calculated cluster centers
print("\nFinal Cluster Centers (V):")
if kfcm_instance.V is not None:
    for i, center in enumerate(kfcm_instance.V):
        center_str = ", ".join([f"{coord:.2f}" for coord in center])
        print(f"  Cluster {i}: [{center_str}]")

# Group points by their predicted cluster for clear output
clustered_results = {i: [] for i in range(kfcm_instance.n_clusters)}
for i in range(X.shape[0]):
    label = predicted_labels[i]
    point = X[i]
    clustered_results[label].append(point)

print("\nData Points Grouped by Predicted Cluster:")
for label, points in clustered_results.items():
    print(f"\n  --- Cluster {label} ---")
    for point in points:
        point_str = ", ".join([f"{coord:.2f}" for coord in point])
        print(f"    Point [{point_str}]")
        
print("\n--- Test Run Finished ---")
